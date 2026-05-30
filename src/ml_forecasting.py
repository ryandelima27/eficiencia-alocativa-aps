"""
================================================================================
CAMADA 5 — PREVISÃO DE DEMANDA COM GATE DE CONFIABILIDADE
Arcabouço de alocação de estoques sob demanda incerta
================================================================================

Autor  : Ryan Gabriel de Lima Nascimento
Versão : 2.0.0
Data   : Maio/2026

DESCRIÇÃO
---------
Previsão de demanda pooled (painel inteiro, não par a par) com gate de
confiabilidade que escala com T (meses de histórico acumulado). Integra-se
ao pipeline como Camada 5, alimentando a Camada 4 com demanda prevista quando
confiabilidade >= 'preliminar'.

Modelos candidatos em ordem de complexidade crescente:
    1. Baseline: média móvel ponderada (MMP) da Camada 3 — piso obrigatório.
       Nenhum ML é promovido se não vencer este baseline (DOC-PROJ §10.5.3).
    2. Ridge: regressão regularizada via TimeSeriesSplit CV (scikit-learn).
       Disponível com T >= T_MIN_PRELIMINAR (default 8).
    3. LightGBM: gradient boosting sobre o painel pooled.
       Captura não-linearidades; mesmos requisitos do Ridge.
    4. Poisson GLM: tecnicamente correto para dados de contagem.
       Preferível para pares com μ < 5 (statsmodels).

GATE DE CONFIABILIDADE (DOC-PROJ §10.5.1)
------------------------------------------
    T < 8   → 'insuficiente': modo sombra, NÃO substitui Camadas 3-4.
    T < 18  → 'preliminar':   previsões ativas, sem sazonalidade.
    T >= 18 → 'operacional':  sazonalidade habilitada, ciclos completos.

Os cortes são parâmetros (MLConfig), não hardcoded.

DEPENDÊNCIAS OPCIONAIS
----------------------
    scikit-learn >= 1.3   (Ridge, TimeSeriesSplit, StandardScaler)
    lightgbm     >= 4.0   (LGBMRegressor)
    statsmodels  >= 0.14  (GLM Poisson)

Referência: DOC-PROJ §10.5
================================================================================
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SklearnPipeline
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    _LGBM = False

try:
    import statsmodels.api as sm
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False

SEED = 20260501  # Sincronizado com pipeline principal (DOC-PROJ §7)


# ============================================================================
# GATE DE CONFIABILIDADE (DOC-PROJ §10.5.1)
# ============================================================================

def gate_confiabilidade(T: int, t_preliminar: int, t_operacional: int) -> str:
    """
    Classifica o nível de confiabilidade em função de T (meses acumulados).

    Parameters
    ----------
    T : int
        Número de meses completos no histórico acumulado.
    t_preliminar : int
        Corte mínimo para 'preliminar' (MLConfig.T_MIN_PRELIMINAR).
    t_operacional : int
        Corte para 'operacional' + sazonalidade (MLConfig.T_OPERACIONAL).

    Returns
    -------
    str : 'insuficiente' | 'preliminar' | 'operacional'

    Notes
    -----
    Com confiabilidade='insuficiente' o módulo opera em modo sombra: gera
    previsões e acumula erro, mas NÃO substitui as políticas das Camadas 3-4.
    """
    if T < t_preliminar:
        return 'insuficiente'
    if T < t_operacional:
        return 'preliminar'
    return 'operacional'


# ============================================================================
# ENGENHARIA DE FEATURES (DOC-PROJ §10.5.2)
# ============================================================================

def construir_features(
    historico: pd.DataFrame,
    metricas_xyz: pd.DataFrame,
    incluir_sazonalidade: bool = False,
    n_lags: int = 4,
    incluir_proximo_periodo: bool = False,
) -> pd.DataFrame:
    """
    Constrói a matriz de features para o painel histórico.

    Cada linha representa (unidade, produto, mes). Variável-alvo: consumo_total.
    Quando incluir_proximo_periodo=True, acrescenta uma linha por par com
    mes=T+1 e consumo_total=NaN — usada para previsão fora da amostra.

    Features construídas
    --------------------
    Temporais:
        lag_1..lag_n : consumo em t-1..t-n (0 quando indisponível, flag _disp)
        media_3m     : média dos 3 meses anteriores (shift + rolling)
        std_3m       : desvio-padrão dos 3 meses anteriores
        trend_idx    : posição ordinal no histórico do par (0, 1, 2, ...)
        mes_num      : número do mês (1–12); sazonalidade só com T>=18

    Categóricas (one-hot):
        cat_*        : dummies de categoria (escritório, copa, limpeza, ...)
        unit_*       : dummies de unidade (15 unidades; one-hot preferível a
                       target-encoding com N=15 e histórico curto)

    Regime (Camada 2):
        regime_X_regular, regime_Y_moderado, regime_Z_erratico,
        regime_baixa_rotatividade, cv_hist, consumo_medio_hist

    Parameters
    ----------
    historico : pd.DataFrame
        Painel acumulado. Colunas mínimas: unidade, produto, categoria, mes,
        consumo_total.
    metricas_xyz : pd.DataFrame
        Saída da Camada 2 — contém classe_xyz, cv, consumo_medio_mensal.
    incluir_sazonalidade : bool
        Habilitar apenas quando T >= T_OPERACIONAL (risco de overfitting antes).
    n_lags : int
        Número de lags temporais (default 4, alinhado com os 4 pesos do MMP).
    incluir_proximo_periodo : bool
        Se True, acrescenta linha sintética mes=T+1 por par para predição.

    Returns
    -------
    pd.DataFrame
        Linhas históricas + (opcionalmente) linhas T+1 com consumo_total=NaN.
    """
    hist = historico.copy()

    # Opcional: acrescenta linha T+1 por par
    if incluir_proximo_periodo:
        max_mes = pd.to_datetime(hist['mes'].max())
        proximo_mes_str = (max_mes + pd.DateOffset(months=1)).strftime('%Y-%m')
        ultimos = (hist.groupby(['unidade', 'produto', 'categoria'])
                       .size().reset_index(name='_')[['unidade', 'produto', 'categoria']])
        ultimos['mes'] = proximo_mes_str
        ultimos['consumo_total'] = np.nan
        # Preenche colunas numéricas adicionais com NaN se existirem
        for col in ['consumo_reg', 'consumo_extra', 'demanda_total']:
            if col in hist.columns:
                ultimos[col] = np.nan
        hist = pd.concat([hist, ultimos], ignore_index=True)

    hist = hist.sort_values(['unidade', 'produto', 'mes']).reset_index(drop=True)

    # --- Lags e estatísticas rolling (por par) ---
    def _engenharia_temporal(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('mes').copy()
        ct = g['consumo_total']
        for i in range(1, n_lags + 1):
            lag = ct.shift(i)
            g[f'lag_{i}_disp'] = (~lag.isna()).astype(int)
            g[f'lag_{i}'] = lag.fillna(0.0)
        g['media_3m'] = ct.shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
        g['std_3m'] = ct.shift(1).rolling(3, min_periods=1).std(ddof=0).fillna(0.0)
        g['trend_idx'] = np.arange(len(g))
        return g

    # Pandas 3.x remove colunas de groupby do resultado de apply — loop explícito
    # preserva todas as colunas incluindo 'unidade' e 'produto'.
    partes = []
    for _, g in hist.groupby(['unidade', 'produto'], sort=False):
        partes.append(_engenharia_temporal(g))
    hist = pd.concat(partes, ignore_index=True)

    # Mês numérico (sazonalidade condicional)
    hist['mes_num'] = pd.to_datetime(hist['mes']).dt.month.astype(int)

    # --- Join com métricas XYZ ---
    xyz_cols = ['unidade', 'produto', 'classe_xyz', 'consumo_medio_mensal', 'cv']
    metricas_sel = metricas_xyz[xyz_cols].copy().rename(
        columns={'consumo_medio_mensal': 'consumo_medio_hist', 'cv': 'cv_hist'}
    )
    hist = hist.merge(metricas_sel, on=['unidade', 'produto'], how='left')

    for regime in ('X_regular', 'Y_moderado', 'Z_erratico', 'baixa_rotatividade'):
        hist[f'regime_{regime}'] = (hist['classe_xyz'] == regime).astype(int)

    hist['cv_hist'] = hist['cv_hist'].fillna(0.0)
    hist['consumo_medio_hist'] = hist['consumo_medio_hist'].fillna(0.0)

    # --- Dummies one-hot ---
    cat_dummies = pd.get_dummies(hist['categoria'], prefix='cat', dtype=int)
    unit_dummies = pd.get_dummies(hist['unidade'], prefix='unit', dtype=int)
    hist = pd.concat([hist, cat_dummies, unit_dummies], axis=1)

    if incluir_sazonalidade:
        mes_dummies = pd.get_dummies(hist['mes_num'], prefix='mes', dtype=int)
        hist = pd.concat([hist, mes_dummies], axis=1)

    return hist


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Retorna as colunas de feature (exclui metadados, target e flags de lag).

    As flags _disp não são features de valor — indicam apenas disponibilidade
    do lag para o baseline MMP; incluí-las como features introduziria leakage
    temporal implícito (o modelo aprenderia quando há menos histórico).
    """
    excluir = {
        'unidade', 'produto', 'categoria', 'mes', 'classe_xyz',
        'consumo_total', 'consumo_reg', 'consumo_extra', 'demanda_total',
    }
    return [c for c in df.columns
            if c not in excluir and not c.endswith('_disp')]


# ============================================================================
# MODELOS (DOC-PROJ §10.5.3)
# ============================================================================

class ForecastModel(ABC):
    """Interface comum para todos os modelos de previsão."""

    nome: str = 'base'

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ForecastModel':
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class BaselineMMPModel(ForecastModel):
    """
    Média móvel ponderada — replicação exata da Camada 3 (DOC-PROJ §10.3).

    Piso obrigatório de comparação. Nenhum modelo ML é promovido se não
    vencer este baseline no walk-forward (DOC-PROJ §10.5.3.1).

    Pesos (0.10, 0.20, 0.30, 0.40): do mais antigo ao mais recente.
    Com menos de 4 lags disponíveis, usa média simples dos lags presentes.
    """
    nome = 'baseline_mmp'

    def __init__(self, pesos: tuple = (0.10, 0.20, 0.30, 0.40)):
        self.pesos = np.array(pesos)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineMMPModel':
        return self  # MMP não tem parâmetros a estimar

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Lags do mais antigo ao mais recente: lag_4, lag_3, lag_2, lag_1
        n = len(self.pesos)
        lag_cols = [f'lag_{i}' for i in range(n, 0, -1) if f'lag_{i}' in X.columns]
        if not lag_cols:
            return X['media_3m'].fillna(0).values if 'media_3m' in X.columns else np.zeros(len(X))
        mat = X[lag_cols].fillna(0).values
        if mat.shape[1] == n:
            return mat @ self.pesos
        # Menos de 4 lags: média simples (alinhado com fallback da Camada 3)
        return mat @ (np.ones(mat.shape[1]) / mat.shape[1])


class RidgeModel(ForecastModel):
    """
    Regressão Ridge com α selecionado via TimeSeriesSplit CV.

    Vantagens: estável com features correlacionadas (lags), baixa variância.
    Limitação: linearidade e suposição gaussiana inadequadas para volumes muito
    baixos — nesses pares, o PoissonModel é preferível.
    Disponível: T >= T_MIN_PRELIMINAR (desabilitado em modo sombra).
    """
    nome = 'ridge'

    def __init__(self, alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0), n_splits: int = 3):
        if not _SKLEARN:
            raise ImportError('scikit-learn não encontrado. pip install scikit-learn')
        self.alphas = alphas
        self.n_splits = n_splits
        self._pipeline: Optional[SklearnPipeline] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RidgeModel':
        n_folds = min(self.n_splits, max(2, len(X) // 10))
        tscv = TimeSeriesSplit(n_splits=n_folds)
        self._pipeline = SklearnPipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=self.alphas, cv=tscv)),
        ])
        self._pipeline.fit(X.fillna(0).astype(float), np.maximum(y.fillna(0), 0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self._pipeline.predict(X.fillna(0).astype(float))
        return np.maximum(preds, 0.0)


class LGBMModel(ForecastModel):
    """
    Gradient boosting (LightGBM) sobre o painel pooled.

    Captura não-linearidades e interações feature×unidade sem especificação
    manual. Parâmetros conservadores (num_leaves=15) enquanto T < 18 para
    evitar overfitting com histórico curto.

    Objetivo L1 (MAE) preferível a L2 (RMSE) por ser mais robusto a picos
    de pedidos extras esporádicos.
    Disponível: T >= T_MIN_PRELIMINAR.
    """
    nome = 'lgbm'

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05,
                 num_leaves: int = 15, min_child_samples: int = 10):
        if not _LGBM:
            raise ImportError('lightgbm não encontrado. pip install lightgbm')
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=SEED,
            verbose=-1,
            objective='regression_l1',
        )
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LGBMModel':
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(X.fillna(0).astype(float), np.maximum(y.fillna(0), 0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self._model.predict(X.fillna(0).astype(float)), 0.0)


class PoissonModel(ForecastModel):
    """
    GLM Poisson via statsmodels — correto para dados de contagem inteiros.

    Garante previsões não-negativas por construção e modela a variância
    proporcional à média (Var = μ), mais realista que o modelo gaussiano
    para itens de baixo volume (μ < 5).

    Limitação: GLM Poisson não converge com muitas features em amostras
    pequenas. Quando n_obs < threshold, usa apenas lags + trend (features
    reduzidas). Documentado como trade-off, não como bug.
    Disponível: T >= T_MIN_PRELIMINAR.
    """
    nome = 'poisson'

    def __init__(self, threshold_features_reduzidas: int = 50):
        if not _STATSMODELS:
            raise ImportError('statsmodels não encontrado. pip install statsmodels')
        self.threshold = threshold_features_reduzidas
        self._result = None
        self._cols: list[str] = []
        self._fallback_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PoissonModel':
        if len(X) < self.threshold:
            self._cols = [c for c in X.columns if c.startswith('lag_') or c == 'trend_idx']
        else:
            self._cols = list(X.columns)

        X_sel = X[self._cols].fillna(0).astype(float) if self._cols else pd.DataFrame(index=X.index)
        X_sm = sm.add_constant(X_sel, has_constant='add')
        y_clean = np.maximum(y.fillna(0).values, 0)
        self._fallback_mean = float(y_clean.mean())

        try:
            glm = sm.GLM(y_clean, X_sm, family=sm.families.Poisson())
            self._result = glm.fit(disp=False)
        except Exception:
            self._result = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._result is None:
            return np.full(len(X), self._fallback_mean)
        X_sel = X[self._cols].fillna(0).astype(float) if self._cols else pd.DataFrame(index=X.index)
        X_sm = sm.add_constant(X_sel, has_constant='add')
        # Alinha colunas que possam ter ficado de fora na predição
        for c in self._result.model.exog_names:
            if c not in X_sm.columns:
                X_sm[c] = 0.0
        X_sm = X_sm[self._result.model.exog_names]
        return np.maximum(self._result.predict(X_sm).values, 0.0)


# ============================================================================
# FACTORY DE MODELOS
# ============================================================================

def _criar_modelo(nome: str, ridge_alphas: tuple) -> ForecastModel:
    """Instancia um modelo pelo nome com configuração padrão."""
    if nome == 'baseline_mmp':
        return BaselineMMPModel()
    if nome == 'ridge':
        return RidgeModel(alphas=ridge_alphas)
    if nome == 'lgbm':
        return LGBMModel()
    if nome == 'poisson':
        return PoissonModel()
    raise ValueError(f'Modelo desconhecido: {nome}')


def get_modelos_disponiveis(T: int, t_preliminar: int, ridge_alphas: tuple) -> list[ForecastModel]:
    """
    Retorna lista de modelos a avaliar dado T.

    Baseline sempre presente. Modelos ML condicionais a T >= t_preliminar
    e à disponibilidade das dependências opcionais.
    """
    modelos: list[ForecastModel] = [BaselineMMPModel()]
    if T < t_preliminar:
        return modelos  # Modo sombra: apenas baseline
    if _SKLEARN:
        modelos.append(RidgeModel(alphas=ridge_alphas))
    if _LGBM:
        modelos.append(LGBMModel())
    if _STATSMODELS:
        modelos.append(PoissonModel())
    return modelos


# ============================================================================
# CAMADA 5 — ORQUESTRADOR (DOC-PROJ §10.5)
# ============================================================================

def camada5_prever(
    historico: pd.DataFrame,
    metricas_xyz: pd.DataFrame,
    t_preliminar: int,
    t_operacional: int,
    ridge_alphas: tuple,
    modelo_campeao: Optional[str],
    output_dir: Path,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Orquestra a Camada 5: treina no histórico completo e gera previsões
    para T+1 com flag de confiabilidade por par.

    Parameters
    ----------
    historico : pd.DataFrame
        Painel acumulado (saída do IngestorIncremental).
    metricas_xyz : pd.DataFrame
        Métricas por par da Camada 2 (classe_xyz, cv, consumo_medio_mensal).
    t_preliminar, t_operacional : int
        Cortes de confiabilidade (MLConfig).
    ridge_alphas : tuple
        Candidatos de α para seleção do Ridge via CV.
    modelo_campeao : str | None
        Nome do modelo promovido no ModelRegistry; usa baseline se None.
    output_dir : Path
        Onde salvar 05_previsoes.csv.
    log : Logger | None

    Returns
    -------
    pd.DataFrame
        Colunas: unidade, produto, categoria, mes_previsao, previsao,
                 intervalo_inferior_90, intervalo_superior_90,
                 confiabilidade, modelo_usado, previsao_substitui_c4.

    Notes
    -----
    Intervalo de predição: bootstrap não-paramétrico sobre resíduos de treino
    (5%–95%). Supõe resíduos i.i.d., o que é uma aproximação em série temporal.
    Declarado como limitação (DOC-PROJ §14).
    """
    if log is None:
        log = logging.getLogger(__name__)

    T = historico['mes'].nunique()
    confiab = gate_confiabilidade(T, t_preliminar, t_operacional)
    incluir_saz = confiab == 'operacional'

    log.info(f'Camada 5 — T={T} meses | confiabilidade={confiab!r} | '
             f'sazonalidade={"sim" if incluir_saz else "não"}')

    modelos_disp = get_modelos_disponiveis(T, t_preliminar, ridge_alphas)
    log.info(f'  → modelos disponíveis: {[m.nome for m in modelos_disp]}')

    # Escolhe modelo (campeão do registry ou baseline)
    modelo_em_uso = modelos_disp[0]
    if modelo_campeao:
        for m in modelos_disp:
            if m.nome == modelo_campeao:
                modelo_em_uso = m
                break

    log.info(f'  → modelo em uso: {modelo_em_uso.nome}')

    # Constrói features de treino (histórico sem T+1)
    feat_treino = construir_features(historico, metricas_xyz, incluir_saz, n_lags=4,
                                     incluir_proximo_periodo=False)
    feat_cols = get_feature_cols(feat_treino)

    X_treino = feat_treino[feat_cols].fillna(0).astype(float)
    y_treino = feat_treino['consumo_total'].fillna(0)

    modelo_em_uso.fit(X_treino, y_treino)

    # Constrói features incluindo T+1 para predição
    feat_completo = construir_features(historico, metricas_xyz, incluir_saz, n_lags=4,
                                       incluir_proximo_periodo=True)
    feat_prox = feat_completo[feat_completo['consumo_total'].isna()].copy()

    # Alinha colunas: adiciona com 0 as que surgiram no treino mas não em T+1
    X_pred = feat_prox.reindex(columns=feat_cols, fill_value=0).astype(float)

    previsoes_raw = np.maximum(modelo_em_uso.predict(X_pred), 0.0).astype(float)

    # Intervalo bootstrap sobre resíduos de treino (aproximação i.i.d.)
    rng = np.random.default_rng(SEED)
    residuos = (y_treino.values.astype(float) - modelo_em_uso.predict(X_treino).astype(float))
    n_boot = 500
    boot = previsoes_raw[:, None] + rng.choice(residuos.astype(float),
                                                size=(len(previsoes_raw), n_boot),
                                                replace=True)
    boot = np.maximum(boot.astype(float), 0.0)
    inf_90 = np.percentile(boot, 5, axis=1).astype(float)
    sup_90 = np.percentile(boot, 95, axis=1).astype(float)

    # Próximo mês calendário
    ultimo_mes = pd.to_datetime(historico['mes'].max())
    prox_mes = (ultimo_mes + pd.DateOffset(months=1)).strftime('%Y-%m')

    resultado = feat_prox[['unidade', 'produto', 'categoria']].copy().reset_index(drop=True)
    resultado['mes_previsao'] = prox_mes
    resultado['previsao'] = np.round(previsoes_raw, 2)
    resultado['intervalo_inferior_90'] = np.round(inf_90, 2)
    resultado['intervalo_superior_90'] = np.round(sup_90, 2)
    resultado['confiabilidade'] = confiab
    resultado['modelo_usado'] = modelo_em_uso.nome
    resultado['previsao_substitui_c4'] = confiab != 'insuficiente'

    resultado.to_csv(output_dir / '05_previsoes.csv', index=False, encoding='utf-8')
    log.info(f'  → {len(resultado)} previsões para {prox_mes} | salvo: 05_previsoes.csv')

    return resultado
