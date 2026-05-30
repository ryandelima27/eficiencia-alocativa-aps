"""
================================================================================
CAMADA 6 — BACKTESTING WALK-FORWARD, RETREINO VERSIONADO E DETECÇÃO DE DRIFT
Arcabouço de alocação de estoques sob demanda incerta
================================================================================

Autor  : Ryan Gabriel de Lima Nascimento
Versão : 2.0.0
Data   : Maio/2026

DESCRIÇÃO
---------
Coração da honestidade do módulo ML. Implementa:

    1. IngestorIncremental  — persistência e deduplicação do histórico mensal
    2. WalkForwardValidator — backtest sem data leakage (treina [1..t], prevê t+1)
    3. ModelRegistry        — versiona modelos e registra promoções
    4. DriftMonitor         — detecta regime shift via z-score do erro recente
    5. fechar_ciclo         — substitui demanda empírica pela prevista na Camada 4
    6. camada6_retreinar    — orquestra tudo e determina o campeão ativo

POLÍTICA DE PROMOÇÃO (DOC-PROJ §10.6.3)
-----------------------------------------
Um challenger (modelo ML) é promovido somente se:
    1. Vencer o baseline (MMP) em MAE no backtest walk-forward, E
    2. Vencer o campeão atual (ou não existir campeão anterior).

Essa política garante que o sistema nunca regride abaixo da heurística
determinística da Camada 3, que é o mínimo aceitável.

DETECÇÃO DE DRIFT (DOC-PROJ §10.6.4)
--------------------------------------
z-score do MAE médio dos últimos 'janela' meses vs. histórico completo.
Flag recalibrar=True quando |z| > z_threshold. Uma implementação Page-Hinkley
mais rigorosa pode substituir isso quando T >= 18.

Referência: DOC-PROJ §10.6
================================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_forecasting import (
    ForecastModel,
    BaselineMMPModel,
    construir_features,
    get_feature_cols,
    get_modelos_disponiveis,
    gate_confiabilidade,
    _criar_modelo,
)

SEED = 20260501  # DOC-PROJ §7


# ============================================================================
# INGESTÃO INCREMENTAL (DOC-PROJ §10.6.1)
# ============================================================================

class IngestorIncremental:
    """
    Mantém historico_mensal.csv — painel acumulado que cresce a cada execução.

    Chave de deduplicação: (unidade, produto, mes).
    Em conflito, a linha mais recente sobrescreve (permite recalcular meses
    parciais sem duplicação).

    O arquivo é o único componente stateful do pipeline. Sem ele, o sistema
    opera como stateless (modo sombra apenas com os 4 meses atuais).
    """

    CHAVE = ['unidade', 'produto', 'mes']
    COLUNAS = ['unidade', 'produto', 'categoria', 'mes',
               'consumo_total', 'consumo_reg', 'consumo_extra', 'demanda_total']

    def __init__(self, caminho: Path):
        self.caminho = caminho

    def load(self) -> pd.DataFrame:
        """Carrega o histórico ou retorna DataFrame vazio."""
        if self.caminho.exists():
            return pd.read_csv(self.caminho, encoding='utf-8')
        return pd.DataFrame(columns=self.COLUNAS)

    def append(self, novo_painel: pd.DataFrame,
                log: Optional[logging.Logger] = None) -> pd.DataFrame:
        """
        Incorpora novo_painel ao histórico com deduplicação por chave.

        Parameters
        ----------
        novo_painel : pd.DataFrame
            Saída da Camada 2 (piv_full). Colunas necessárias: COLUNAS acima.

        Returns
        -------
        pd.DataFrame : histórico atualizado, persistido em self.caminho.
        """
        if log is None:
            log = logging.getLogger(__name__)

        colunas_presentes = [c for c in self.COLUNAS if c in novo_painel.columns]
        novo = novo_painel[colunas_presentes].copy()

        historico_ant = self.load()
        n_antes = len(historico_ant)

        combinado = pd.concat([historico_ant, novo], ignore_index=True)
        # keep='last': linha nova sobrescreve a antiga para o mesmo (u, p, mes)
        combinado = combinado.drop_duplicates(subset=self.CHAVE, keep='last')
        combinado = combinado.sort_values(self.CHAVE).reset_index(drop=True)

        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        combinado.to_csv(self.caminho, index=False, encoding='utf-8')

        T = combinado['mes'].nunique()
        log.info(f'IngestorIncremental — {n_antes} → {len(combinado)} linhas | '
                 f'T={T} meses acumulados')
        return combinado


# ============================================================================
# WALK-FORWARD VALIDATOR (DOC-PROJ §10.6.2)
# ============================================================================

class WalkForwardValidator:
    """
    Backtest walk-forward: treina em [1..t], prevê t+1, avança um passo.

    Nunca usa dado futuro para prever passado — sem data leakage.

    Limitação conhecida: metricas_xyz passadas são calculadas sobre o histórico
    completo (não recalculadas por fold). Isso é uma aproximação aceitável para
    os regimes XYZ (que mudam lentamente), mas deve ser corrigida quando
    T >= 18 e os regimes forem reavaliados periodicamente.
    """

    def __init__(self, min_treino: int = 3):
        self.min_treino = min_treino

    def validate(
        self,
        nome_modelo: str,
        ridge_alphas: tuple,
        historico: pd.DataFrame,
        metricas_xyz: pd.DataFrame,
        t_preliminar: int,
        t_operacional: int,
    ) -> pd.DataFrame:
        """
        Executa validação walk-forward para um modelo.

        Parameters
        ----------
        nome_modelo : str
            Nome do modelo a instanciar em cada fold (via _criar_modelo).
        ridge_alphas : tuple
            Alphas para RidgeModel (ignorado para outros modelos).
        historico : pd.DataFrame
            Painel acumulado completo.
        metricas_xyz : pd.DataFrame
            Métricas XYZ por par (calculadas sobre o histórico completo).
        t_preliminar, t_operacional : int
            Cortes de confiabilidade para decidir sazonalidade por fold.

        Returns
        -------
        pd.DataFrame
            Colunas: fold, mes_test, unidade, produto, actual, predicted,
                     abs_error, sq_error, pct_error.
            Vazio se histórico insuficiente (T < min_treino + 1).
        """
        meses = sorted(historico['mes'].unique())
        T = len(meses)

        if T < self.min_treino + 1:
            return pd.DataFrame()

        registros = []
        for t_idx in range(self.min_treino, T):
            meses_treino = meses[:t_idx]
            mes_test = meses[t_idx]

            train_hist = historico[historico['mes'].isin(meses_treino)].copy()
            test_hist = historico[historico['mes'] == mes_test].copy()

            incluir_saz = gate_confiabilidade(
                len(meses_treino), t_preliminar, t_operacional
            ) == 'operacional'

            # Features de treino
            try:
                feat_treino = construir_features(train_hist, metricas_xyz, incluir_saz,
                                                 n_lags=4, incluir_proximo_periodo=False)
                feat_cols = get_feature_cols(feat_treino)
                X_tr = feat_treino[feat_cols].fillna(0).astype(float)
                y_tr = feat_treino['consumo_total'].fillna(0)

                modelo = _criar_modelo(nome_modelo, ridge_alphas)
                modelo.fit(X_tr, y_tr)

                # Features para T+1 (o mês de teste)
                feat_full = construir_features(train_hist, metricas_xyz, incluir_saz,
                                               n_lags=4, incluir_proximo_periodo=True)
                feat_prox = feat_full[feat_full['consumo_total'].isna()].copy()
                X_pred = feat_prox.reindex(columns=feat_cols, fill_value=0).astype(float)
                preds = np.maximum(modelo.predict(X_pred), 0.0)

                pred_df = feat_prox[['unidade', 'produto']].copy().reset_index(drop=True)
                pred_df['predicted'] = preds

            except Exception:
                continue

            actual_df = test_hist[['unidade', 'produto', 'consumo_total']].rename(
                columns={'consumo_total': 'actual'}
            )
            merged = pred_df.merge(actual_df, on=['unidade', 'produto'], how='inner')

            for _, row in merged.iterrows():
                err = abs(row['actual'] - row['predicted'])
                pct = err / row['actual'] if row['actual'] > 0 else np.nan
                registros.append({
                    'fold': t_idx,
                    'mes_test': mes_test,
                    'unidade': row['unidade'],
                    'produto': row['produto'],
                    'actual': row['actual'],
                    'predicted': row['predicted'],
                    'abs_error': err,
                    'sq_error': err ** 2,
                    'pct_error': pct,
                })

        return pd.DataFrame(registros)

    @staticmethod
    def agregar_metricas(backtest: pd.DataFrame) -> dict:
        """Agrega MAE, RMSE e MAPE do backtest completo."""
        if backtest.empty:
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'n_obs': 0}
        return {
            'mae': float(backtest['abs_error'].mean()),
            'rmse': float(np.sqrt(backtest['sq_error'].mean())),
            'mape': float(backtest['pct_error'].dropna().mean()),
            'n_obs': len(backtest),
        }


# ============================================================================
# MODEL REGISTRY (DOC-PROJ §10.6.3)
# ============================================================================

class ModelRegistry:
    """
    Versiona modelos e registra promoções em model_registry.csv.

    Cada linha do arquivo representa uma avaliação. A coluna 'promovido'
    indica se o modelo passou a ser o campeão ativo naquele momento.

    Política de promoção: challenger é promovido somente se vencer o baseline
    E o campeão atual em MAE. Se não houver campeão, o baseline assume o papel.
    """

    COLUNAS = ['versao', 'data_registro', 'modelo', 'T_treino',
               'mae', 'rmse', 'mape', 'n_obs', 'promovido', 'notas']

    def __init__(self, caminho: Path):
        self.caminho = caminho

    def load(self) -> pd.DataFrame:
        if self.caminho.exists():
            return pd.read_csv(self.caminho, encoding='utf-8')
        return pd.DataFrame(columns=self.COLUNAS)

    def registrar(
        self,
        nome_modelo: str,
        T_treino: int,
        metricas: dict,
        promovido: bool,
        notas: str = '',
    ) -> str:
        """Adiciona entrada e retorna o ID da versão (ex.: 'v0003')."""
        df = self.load()
        versao = f'v{len(df) + 1:04d}'
        nova = pd.DataFrame([{
            'versao': versao,
            'data_registro': datetime.now().isoformat(timespec='seconds'),
            'modelo': nome_modelo,
            'T_treino': T_treino,
            'mae': round(metricas.get('mae', np.nan), 4),
            'rmse': round(metricas.get('rmse', np.nan), 4),
            'mape': round(metricas.get('mape', np.nan), 4),
            'n_obs': metricas.get('n_obs', 0),
            'promovido': promovido,
            'notas': notas,
        }])
        df = pd.concat([df, nova], ignore_index=True)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.caminho, index=False, encoding='utf-8')
        return versao

    def campeao_atual(self) -> Optional[dict]:
        """Retorna o último modelo promovido, ou None se nenhum ainda."""
        df = self.load()
        promovidos = df[df['promovido'].astype(bool)]
        if promovidos.empty:
            return None
        return promovidos.iloc[-1].to_dict()

    def deve_promover(
        self,
        challenger_metricas: dict,
        baseline_metricas: dict,
    ) -> tuple[bool, str]:
        """
        Avalia se o challenger deve ser promovido.

        Regras (ambas obrigatórias):
            1. challenger.mae < baseline.mae
            2. challenger.mae < campeao_atual.mae (ou não há campeão)

        Returns
        -------
        (promover: bool, motivo: str)
        """
        ch_mae = challenger_metricas.get('mae', np.inf)
        bl_mae = baseline_metricas.get('mae', np.inf)

        if np.isnan(ch_mae) or np.isnan(bl_mae):
            return False, 'métricas inválidas (nan) — histórico insuficiente para backtest'

        if ch_mae >= bl_mae:
            return (False,
                    f'não venceu baseline '
                    f'(challenger={ch_mae:.4f} >= baseline={bl_mae:.4f})')

        campeao = self.campeao_atual()
        if campeao:
            champ_mae = float(campeao.get('mae', np.inf))
            if not np.isnan(champ_mae) and ch_mae >= champ_mae:
                return (False,
                        f'não venceu campeão atual '
                        f'(challenger={ch_mae:.4f} >= campeão={champ_mae:.4f})')

        return True, f'venceu baseline ({bl_mae:.4f}) e campeão'


# ============================================================================
# DRIFT MONITOR (DOC-PROJ §10.6.4)
# ============================================================================

class DriftMonitor:
    """
    Detecta mudança estrutural de regime via z-score do erro recente.

    Compara o MAE médio dos últimos `janela` meses contra o histórico de erros
    acumulado. Flag recalibrar=True quando |z| > z_threshold.

    Implementação conservadora: z-score simples. Uma implementação Page-Hinkley
    (mais sensível a mudanças graduais) pode ser adicionada quando T >= 18.

    Limitação: com poucos pontos de erro (T - min_treino < janela + 2), o
    monitor retorna recalibrar=False por padrão — não é possível detectar
    drift sem histórico suficiente.
    """

    def __init__(self, janela: int = 3, z_threshold: float = 2.0):
        self.janela = janela
        self.z_threshold = z_threshold
        self._erros: list[float] = []

    def atualizar(self, erros_fold: list[float]) -> None:
        """Incorpora erros absolutos de um novo fold."""
        self._erros.extend(erros_fold)

    def checar_drift(self) -> dict:
        """
        Verifica drift. Retorna dict com recalibrar, z_score e diagnóstico.

        Returns
        -------
        dict : {recalibrar: bool, z_score: float, media_recente: float,
                media_historica: float, motivo: str}
        """
        if len(self._erros) < self.janela + 2:
            return {
                'recalibrar': False,
                'z_score': np.nan,
                'media_recente': np.nan,
                'media_historica': np.nan,
                'motivo': 'histórico de erros insuficiente para detecção de drift',
            }

        hist = np.array(self._erros)
        recentes = hist[-self.janela:]
        anteriores = hist[:-self.janela]

        media_rec = recentes.mean()
        media_ant = anteriores.mean()
        std_ant = anteriores.std() if len(anteriores) > 1 else 1.0

        z = (media_rec - media_ant) / (std_ant + 1e-9)
        recalibrar = abs(z) > self.z_threshold

        return {
            'recalibrar': recalibrar,
            'z_score': round(float(z), 4),
            'media_recente': round(float(media_rec), 4),
            'media_historica': round(float(media_ant), 4),
            'motivo': f'|z|={abs(z):.2f} {">" if recalibrar else "<="} {self.z_threshold}',
        }


# ============================================================================
# FECHAMENTO DO CICLO (DOC-PROJ §10.6.5)
# ============================================================================

def fechar_ciclo(
    dominancia_df: pd.DataFrame,
    previsoes: pd.DataFrame,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Substitui demanda empírica pela prevista na análise de dominância (Camada 4)
    quando confiabilidade >= 'preliminar'.

    Com confiabilidade='insuficiente' (modo sombra), a Camada 4 permanece
    operando sobre a demanda observada histórica — as previsões existem mas não
    alteram o processo decisório.

    Parameters
    ----------
    dominancia_df : pd.DataFrame
        Saída da Camada 4 (04_dominancia.csv).
    previsoes : pd.DataFrame
        Saída da Camada 5 (05_previsoes.csv) com flag previsao_substitui_c4.

    Returns
    -------
    pd.DataFrame
        dominancia_df enriquecido com: demanda_prevista, fonte_demanda,
        confiabilidade_ml, modelo_ml.
    """
    if log is None:
        log = logging.getLogger(__name__)

    dom = dominancia_df.copy()
    prev_ativas = previsoes[previsoes['previsao_substitui_c4']].copy()

    if prev_ativas.empty:
        log.info('fechar_ciclo — modo sombra ativo; Camada 4 usa demanda empírica')
        dom['demanda_prevista'] = np.nan
        dom['fonte_demanda'] = 'empirica'
        dom['confiabilidade_ml'] = 'insuficiente'
        dom['modelo_ml'] = np.nan
        return dom

    dom = dom.merge(
        prev_ativas[['unidade', 'produto', 'previsao', 'confiabilidade', 'modelo_usado']],
        on=['unidade', 'produto'],
        how='left',
    )
    dom.rename(columns={'previsao': 'demanda_prevista',
                        'confiabilidade': 'confiabilidade_ml',
                        'modelo_usado': 'modelo_ml'}, inplace=True)
    dom['fonte_demanda'] = np.where(
        dom['demanda_prevista'].notna(), 'ml_prevista', 'empirica'
    )

    n_sub = (dom['fonte_demanda'] == 'ml_prevista').sum()
    log.info(f'fechar_ciclo — {n_sub}/{len(dom)} pares com demanda substituída por previsão ML')
    return dom


# ============================================================================
# CAMADA 6 — ORQUESTRADOR (DOC-PROJ §10.6)
# ============================================================================

def camada6_retreinar(
    historico: pd.DataFrame,
    metricas_xyz: pd.DataFrame,
    t_preliminar: int,
    t_operacional: int,
    ridge_alphas: tuple,
    registry: ModelRegistry,
    drift_monitor: DriftMonitor,
    output_dir: Path,
    log: Optional[logging.Logger] = None,
) -> dict:
    """
    Orquestra backtest walk-forward, tentativa de promoção e registro.

    Fluxo
    -----
    1. Valida todos os modelos disponíveis via walk-forward.
    2. Atualiza DriftMonitor com erros do fold mais recente (baseline).
    3. Avalia se algum challenger deve ser promovido sobre o baseline.
    4. Registra todos os modelos no ModelRegistry.
    5. Salva 06_backtest_walkforward.csv.

    Parameters
    ----------
    historico : pd.DataFrame
        Painel acumulado (IngestorIncremental.load()).
    metricas_xyz : pd.DataFrame
        Métricas por par da Camada 2.
    t_preliminar, t_operacional : int
        Cortes de confiabilidade (MLConfig).
    ridge_alphas : tuple
        Candidatos α para RidgeModel.
    registry : ModelRegistry
        Instância persistente entre execuções.
    drift_monitor : DriftMonitor
        Instância persistente entre execuções (acumula histórico de erros).
    output_dir : Path
        Onde salvar o backtest consolidado.
    log : Logger | None

    Returns
    -------
    dict : {campeao: str, metricas_campeao: dict, drift_status: dict,
            modelos_avaliados: list[str]}
    """
    if log is None:
        log = logging.getLogger(__name__)

    T = historico['mes'].nunique()
    confiab = gate_confiabilidade(T, t_preliminar, t_operacional)
    log.info(f'Camada 6 — T={T} | confiabilidade={confiab!r}')

    modelos_disp = get_modelos_disponiveis(T, t_preliminar, ridge_alphas)
    validator = WalkForwardValidator(min_treino=3)

    resultados: dict[str, dict] = {}
    backtest_baseline = pd.DataFrame()

    for modelo in modelos_disp:
        log.info(f'  → backtest walk-forward: {modelo.nome}')
        bt = validator.validate(
            modelo.nome, ridge_alphas, historico, metricas_xyz, t_preliminar, t_operacional
        )
        metricas = WalkForwardValidator.agregar_metricas(bt)
        resultados[modelo.nome] = {'metricas': metricas, 'backtest': bt}
        log.info(f'     MAE={metricas["mae"]:.4f} | RMSE={metricas["rmse"]:.4f} | '
                 f'MAPE={metricas["mape"]:.4f} | n={metricas["n_obs"]}')
        if modelo.nome == 'baseline_mmp':
            backtest_baseline = bt

    # Atualiza DriftMonitor com erros do fold mais recente do baseline
    if not backtest_baseline.empty:
        fold_max = backtest_baseline['fold'].max()
        erros_recentes = backtest_baseline[
            backtest_baseline['fold'] == fold_max
        ]['abs_error'].tolist()
        drift_monitor.atualizar(erros_recentes)

    drift_status = drift_monitor.checar_drift()
    nivel = 'WARNING' if drift_status['recalibrar'] else 'INFO'
    getattr(log, nivel.lower())(
        f'  → drift: {drift_status["motivo"]}'
    )

    # Avaliação de promoção
    bl_metricas = resultados.get('baseline_mmp', {}).get('metricas', {})
    campeao_nome = (registry.campeao_atual() or {}).get('modelo', 'baseline_mmp')

    for modelo in modelos_disp:
        nome = modelo.nome
        ch_metricas = resultados[nome]['metricas']

        if nome == 'baseline_mmp':
            registry.registrar(nome, T, ch_metricas,
                                promovido=False, notas='baseline de referência (não-promovível)')
            continue

        promover, motivo = registry.deve_promover(ch_metricas, bl_metricas)
        registry.registrar(nome, T, ch_metricas, promovido=promover, notas=motivo)

        if promover:
            campeao_nome = nome
            log.info(f'  → PROMOVIDO: {nome} ({motivo})')
        else:
            log.info(f'  → não promovido: {nome} ({motivo})')

    # Salva backtest consolidado
    backtests = []
    for nome, r in resultados.items():
        bt = r['backtest'].copy()
        if not bt.empty:
            bt.insert(0, 'modelo', nome)
            backtests.append(bt)
    if backtests:
        pd.concat(backtests, ignore_index=True).to_csv(
            output_dir / '06_backtest_walkforward.csv', index=False, encoding='utf-8'
        )
        log.info(f'  → backtest consolidado salvo: 06_backtest_walkforward.csv')

    return {
        'campeao': campeao_nome,
        'metricas_campeao': resultados.get(campeao_nome, {}).get('metricas', {}),
        'drift_status': drift_status,
        'modelos_avaliados': list(resultados.keys()),
    }
