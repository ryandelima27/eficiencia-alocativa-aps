"""
================================================================================
PIPELINE DE EFICIÊNCIA ALOCATIVA SOB INCERTEZA ESTOCÁSTICA
Sistema de Materiais de Apoio — Rede Municipal de Atenção Primária à Saúde
================================================================================

Autor          : Ryan Gabriel de Lima Nascimento
Contexto       : Dados operacionais reais, identificações suprimidas (confidencialidade)
Versão         : 1.0.0
Data           : Maio/2026
Licença        : Uso interno (replicação institucional autorizada)

DESCRIÇÃO
---------
Este script implementa o pipeline analítico descrito no documento de projeto
"Modelagem da Eficiência Alocativa de Materiais de Apoio na APS sob
Incerteza Estocástica" (doravante DOC-PROJ §N referencia a seção N do documento).

Arquitetura em 4 camadas independentes mas encadeadas:
    Camada 1 — Consolidação e auditoria do data layer  (DOC-PROJ §10.1)
    Camada 2 — Caracterização descritiva e tipologia   (DOC-PROJ §10.2)
    Camada 3 — Política de cota mensal                  (DOC-PROJ §10.3)
    Camada 4 — Análise de dominância entre políticas    (DOC-PROJ §10.4)

PRINCÍPIOS DE REPLICABILIDADE
-----------------------------
1. Reprodutibilidade absoluta: seeds fixadas (np.random.seed = 20260501).
2. Imutabilidade do input: o arquivo de origem nunca é sobrescrito.
3. Auditabilidade: cada transformação é registrada em log_auditoria.csv com
   contagem de linhas antes/depois e justificativa.
4. Versionamento de parâmetros: todos os parâmetros do modelo estão em
   CONFIG no topo do arquivo. Qualquer alteração deve ser registrada no
   changelog institucional.
5. Validações: asserts em pontos críticos para falhar cedo, não em silêncio.

DEPENDÊNCIAS
------------
    Python  >= 3.10
    pandas  >= 2.0
    numpy   >= 1.24
    openpyxl >= 3.1
    scipy   >= 1.10  (apenas Camada 4)

USO
---
    $ python pipeline_eficiencia_alocativa.py \\
        --input  /caminho/Relatorio_SCM.xls \\
        --output /caminho/saida/

ESTRUTURA DE SAÍDA
------------------
    saida/
    ├── 01_dataset_consolidado.csv         # Camada 1
    ├── 02_metricas_por_par.csv            # Camada 2
    ├── 03_politicas_cota.csv              # Camada 3
    ├── 04_dominancia.csv                  # Camada 4
    ├── log_auditoria.csv                  # Trilha de auditoria
    └── pipeline.log                       # Log de execução
================================================================================
"""

import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURAÇÃO — Todos os parâmetros do modelo concentrados aqui
# ============================================================================

SEED = 20260501  # Fixar para reprodutibilidade (DOC-PROJ §7)

@dataclass(frozen=True)
class Config:
    """Parâmetros do modelo. Alterações requerem registro no changelog."""

    # ---- Camada 1: Consolidação ----
    # Status que indicam pedido válido (não rejeitado)
    STATUS_DEMANDA_VALIDA: tuple = (
        'Finalizado', 'Recebimento Confirmado', 'Enviada',
        'Aprovado', 'Pendente', 'Aguardando Gerente'
    )
    STATUS_ATENDIDO: tuple = ('Finalizado', 'Recebimento Confirmado', 'Enviada')
    STATUS_PENDENTE: tuple = ('Pendente', 'Aprovado', 'Aguardando Gerente')

    # ---- Camada 2: Caracterização ----
    # Cortes de CV para classificação XYZ (DOC-PROJ §10.2.3)
    CV_CORTE_REGULAR: float = 0.30   # CV < 0.30 → regime X (regular)
    CV_CORTE_MODERADO: float = 0.70  # 0.30 <= CV < 0.70 → regime Y (moderado)
    # CV >= 0.70 → regime Z (errático)

    # Filtro de materialidade: pares com consumo médio < este valor saem da
    # análise (baixa rotatividade — necessidade marginal vai ao Anexo)
    CONSUMO_MEDIO_MIN: float = 0.5

    # ---- Camada 3: Política de cota ----
    # Pesos da média móvel ponderada (do mais antigo ao mais recente)
    # Soma deve ser 1.0
    PESOS_MMP: tuple = (0.10, 0.20, 0.30, 0.40)

    # Função k(CV, μ) — fator multiplicador do desvio-padrão (DOC-PROJ §10.3.2)
    K_BAIXA_ROTATIVIDADE: float = 0.5   # μ < 1.0 ou CV < CV_CORTE_REGULAR
    K_MODERADO: float = 1.0             # CV_CORTE_REGULAR <= CV < CV_CORTE_MODERADO
    K_ERRATICO_BAIXO: float = 1.25      # CV_CORTE_MODERADO <= CV < 1.20
    K_ERRATICO_ALTO: float = 1.50       # CV >= 1.20

    # ---- Camada 4: Dominância estocástica ----
    # Número de réplicas bootstrap para construir CDFs simuladas
    N_BOOTSTRAP: int = 5000
    # Nível de significância para teste de dominância
    ALPHA_DOMINANCIA: float = 0.05

    # ---- Período de análise ----
    # Apenas meses completos entram nas médias. Maio/2026 é parcial.
    MESES_REFERENCIA: tuple = ('2026-01', '2026-02', '2026-03', '2026-04')


CONFIG = Config()


@dataclass(frozen=True)
class MLConfig:
    """
    Parâmetros do módulo de Machine Learning (Camadas 5-6).

    Cortes de confiabilidade e hiperparâmetros separados do Config principal
    para deixar explícito que são parâmetros do módulo de ML, não do pipeline
    estatístico clássico. Alterações requerem registro no changelog.
    """

    # Gate de confiabilidade (DOC-PROJ §10.5.1)
    T_MIN_PRELIMINAR: int = 8    # meses mínimos para confiabilidade 'preliminar'
    T_OPERACIONAL: int = 18      # meses para 'operacional' + sazonalidade habilitada

    # Walk-forward (DOC-PROJ §10.6.2)
    WALK_FORWARD_MIN_TREINO: int = 3  # folds mínimos de treino antes de prever

    # Ridge (DOC-PROJ §10.5.3)
    RIDGE_ALPHAS: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)

    # Detecção de drift (DOC-PROJ §10.6.4)
    DRIFT_JANELA: int = 3          # últimos N meses para z-score do erro
    DRIFT_Z_THRESHOLD: float = 2.0  # |z| acima do qual dispara flag de recalibração


ML_CONFIG = MLConfig()


# ============================================================================
# LOGGING E AUDITORIA
# ============================================================================

def configurar_logging(output_dir: Path) -> logging.Logger:
    """Configura logger duplo: console + arquivo."""
    log_file = output_dir / 'pipeline.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class AuditTrail:
    """Registra cada transformação com contagens antes/depois."""

    def __init__(self, output_dir: Path):
        self.records = []
        self.output_dir = output_dir

    def log(self, step: str, n_in: int, n_out: int, rationale: str):
        self.records.append({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'etapa': step,
            'linhas_entrada': n_in,
            'linhas_saida': n_out,
            'delta': n_out - n_in,
            'justificativa': rationale
        })

    def save(self):
        pd.DataFrame(self.records).to_csv(
            self.output_dir / 'log_auditoria.csv',
            index=False, encoding='utf-8'
        )


# ============================================================================
# CAMADA 1 — CONSOLIDAÇÃO E AUDITORIA DO DATA LAYER
# Referência: DOC-PROJ §10.1
# ============================================================================

def camada1_carregar_e_normalizar(input_path: Path,
                                   audit: AuditTrail,
                                   log: logging.Logger) -> pd.DataFrame:
    """
    Carrega o relatório bruto (HTML salvo como .xls) e normaliza.

    Parameters
    ----------
    input_path : Path
        Caminho do arquivo de origem.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas padronizadas e tipos corretos.
    """
    log.info(f"Camada 1.1 — Carregando {input_path}")

    # O relatório do SCM exporta como HTML com extensão .xls — leitura via pd.read_html
    tables = pd.read_html(input_path)
    assert len(tables) == 1, f"Esperava 1 tabela, encontrei {len(tables)}"
    df = tables[0]
    n_inicial = len(df)

    # Renomear para snake_case (DOC-PROJ §10.1.1)
    df.columns = ['unidade', 'data_solic', 'produto', 'categoria',
                  'qtd_solic', 'qtd_env', 'qtd_rec', 'status',
                  'pedido_extra', 'mes_ref']

    # Tipagem
    df['data_solic'] = pd.to_datetime(df['data_solic'])
    df['mes'] = df['data_solic'].dt.to_period('M').astype(str)
    df['qtd_solic'] = pd.to_numeric(df['qtd_solic'], errors='coerce').fillna(0)
    df['qtd_env']   = pd.to_numeric(df['qtd_env'], errors='coerce').fillna(0)
    df['qtd_rec']   = pd.to_numeric(df['qtd_rec'], errors='coerce').fillna(0)

    # Normalização defensiva de categoria (corrige Ó vs O, capitalização)
    df['categoria'] = df['categoria'].str.upper().str.replace('Ó', 'O').str.strip()

    # Converter pedido_extra para booleano
    df['pedido_extra'] = df['pedido_extra'].map({'Sim': True, 'Não': False})

    audit.log('1.1_carga_e_normalizacao', n_inicial, len(df),
              f'Carga inicial do relatório SCM ({input_path.name})')
    log.info(f"  → {len(df)} linhas, {df['unidade'].nunique()} unidades, "
             f"{df['produto'].nunique()} produtos, "
             f"período {df['mes'].min()} a {df['mes'].max()}")
    return df


def camada1_anonimizar(df: pd.DataFrame,
                       audit: AuditTrail,
                       log: logging.Logger,
                       ativo: bool = False) -> pd.DataFrame:
    """
    Anonimiza nomes de unidade para publicação externa (opcional).

    Quando ativo=True, remapeia cada unidade para 'Unidade A', 'Unidade B', ...
    em ordem decrescente de volume de pedidos. Salva o mapa em
    'mapa_anonimizacao.json' para auditoria reversível interna.

    Use ativo=True apenas para gerar artefatos de portfolio/publicação;
    mantenha ativo=False para uso operacional interno.
    """
    if not ativo:
        return df

    log.info("Camada 1.0 — Anonimização de unidades (modo publicação)")
    import json
    ordem = df.groupby('unidade').size().sort_values(ascending=False).index.tolist()
    mapa = {nome: f'Unidade {chr(65 + i)}' for i, nome in enumerate(ordem)}
    df['unidade'] = df['unidade'].map(mapa)
    with open('mapa_anonimizacao.json', 'w', encoding='utf-8') as f:
        json.dump(mapa, f, ensure_ascii=False, indent=2)
    audit.log('1.0_anonimizacao', len(df), len(df),
              f'{len(mapa)} unidades remapeadas (modo publicação)')
    log.info(f"  → {len(mapa)} unidades anonimizadas")
    return df


def camada1_flags_qualidade(df: pd.DataFrame,
                             audit: AuditTrail,
                             log: logging.Logger) -> pd.DataFrame:
    """
    Detecta e flagga erros de lançamento (DOC-PROJ §10.1.2).

    Tipos de erro tratados:
        A. qtd_env > qtd_solic  (entrega excede pedido)
        B. qtd_rec > qtd_env    (recebimento excede envio)
        C. status finalizado/confirmado com qtd_rec = 0 (subnotificação)
    """
    log.info("Camada 1.2 — Auditoria de qualidade")

    df['flag_env_maior_solic'] = df['qtd_env'] > df['qtd_solic']
    df['flag_rec_maior_env']   = df['qtd_rec'] > df['qtd_env']
    df['flag_fin_sem_rec'] = (
        df['status'].isin(['Finalizado', 'Recebimento Confirmado'])
        & (df['qtd_rec'] == 0)
    )
    df['flag_qualquer_erro'] = (
        df['flag_env_maior_solic']
        | df['flag_rec_maior_env']
        | df['flag_fin_sem_rec']
    )

    log.info(f"  → Erros detectados: env>solic={df['flag_env_maior_solic'].sum()}, "
             f"rec>env={df['flag_rec_maior_env'].sum()}, "
             f"fin sem rec={df['flag_fin_sem_rec'].sum()} "
             f"(total: {df['flag_qualquer_erro'].sum()}, "
             f"{df['flag_qualquer_erro'].mean()*100:.1f}%)")
    return df


def camada1_consumo_consolidado(df: pd.DataFrame,
                                  audit: AuditTrail,
                                  log: logging.Logger) -> pd.DataFrame:
    """
    Constrói qtd_consumo como variável-alvo consolidada (DOC-PROJ §10.1.3).

    Árvore de decisão hierárquica:
        1. status = Rejeitado            → qtd_consumo = 0
        2. qtd_rec > 0                    → qtd_consumo = min(qtd_rec, qtd_env)
        3. qtd_rec = 0 e qtd_env > 0      → qtd_consumo = min(qtd_env, qtd_solic)
        4. qtd_rec = qtd_env = 0 e
           status em STATUS_PENDENTE      → qtd_consumo = qtd_solic
        5. caso contrário                 → qtd_consumo = 0

    Justificativa: contorna a subnotificação de qtd_rec (~9% dos casos) sem
    superestimar via qtd_solic em pedidos rejeitados.
    """
    log.info("Camada 1.3 — Consolidação da variável-alvo qtd_consumo")

    def regra(row) -> float:
        if row['status'] == 'Rejeitado':
            return 0.0
        if row['qtd_rec'] > 0:
            return min(row['qtd_rec'], row['qtd_env']) if row['qtd_env'] > 0 else row['qtd_rec']
        if row['qtd_env'] > 0:
            return min(row['qtd_env'], row['qtd_solic']) if row['qtd_solic'] > 0 else row['qtd_env']
        if row['status'] in CONFIG.STATUS_PENDENTE:
            return row['qtd_solic']
        return 0.0

    df['qtd_consumo'] = df.apply(regra, axis=1)
    df['qtd_demanda'] = np.where(df['status'] == 'Rejeitado', 0, df['qtd_solic'])

    audit.log('1.3_consolidacao_alvo', len(df), len(df),
              'Construção de qtd_consumo via árvore hierárquica rec > env > solic')

    log.info(f"  → consumo total no período: {df['qtd_consumo'].sum():.0f} unidades")
    return df


# ============================================================================
# CAMADA 2 — CARACTERIZAÇÃO DESCRITIVA E TIPOLOGIA XYZ
# Referência: DOC-PROJ §10.2
# ============================================================================

def camada2_agregar_mensal(df: pd.DataFrame,
                            audit: AuditTrail,
                            log: logging.Logger) -> pd.DataFrame:
    """
    Agrega por (unidade, produto, mês, tipo de pedido) e pivota.

    Saída: linha = (unidade, produto, categoria, mes); colunas decompondo
    regular vs extra para solicitada, enviada, recebida, consumo, demanda.
    """
    log.info("Camada 2.1 — Agregação mensal")

    dff = df[df['mes'].isin(CONFIG.MESES_REFERENCIA)]

    agg = (dff.groupby(['unidade', 'produto', 'categoria', 'mes', 'pedido_extra'])
              .agg(solic=('qtd_solic', 'sum'),
                   env=('qtd_env', 'sum'),
                   rec=('qtd_rec', 'sum'),
                   consumo=('qtd_consumo', 'sum'),
                   demanda=('qtd_demanda', 'sum'),
                   n_pedidos=('qtd_solic', 'count'))
              .reset_index())

    piv = agg.pivot_table(
        index=['unidade', 'produto', 'categoria', 'mes'],
        columns='pedido_extra',
        values=['solic', 'env', 'rec', 'consumo', 'demanda', 'n_pedidos'],
        fill_value=0
    ).reset_index()
    piv.columns = ['_'.join([str(c) for c in col]).strip('_')
                    .replace('True', 'extra').replace('False', 'reg')
                   for col in piv.columns]

    # Garantir todas as colunas (pode faltar se não houver extras/regulares)
    for tipo in ('reg', 'extra'):
        for m in ('solic', 'env', 'rec', 'consumo', 'demanda', 'n_pedidos'):
            c = f'{m}_{tipo}'
            if c not in piv.columns:
                piv[c] = 0

    # Totais derivados
    piv['solic_total']    = piv['solic_reg'] + piv['solic_extra']
    piv['consumo_total']  = piv['consumo_reg'] + piv['consumo_extra']
    piv['demanda_total']  = piv['demanda_reg'] + piv['demanda_extra']
    piv['gap_atendimento'] = piv['demanda_total'] - piv['consumo_total']

    # Painel balanceado: criar linhas com zeros para meses sem pedido
    base = (dff.groupby(['unidade', 'produto', 'categoria']).size()
                .reset_index(name='_')[['unidade', 'produto', 'categoria']])
    matriz = base.merge(
        pd.DataFrame({'mes': list(CONFIG.MESES_REFERENCIA)}),
        how='cross'
    )
    piv_full = matriz.merge(piv,
                             on=['unidade', 'produto', 'categoria', 'mes'],
                             how='left').fillna(0)

    audit.log('2.1_agregacao_mensal', len(df), len(piv_full),
              f'Pivot mensal balanceado: {piv_full.shape}')
    log.info(f"  → painel balanceado: {len(piv_full)} linhas "
             f"({piv_full[['unidade','produto']].drop_duplicates().shape[0]} pares × "
             f"{len(CONFIG.MESES_REFERENCIA)} meses)")
    return piv_full


def camada2_metricas_por_par(piv: pd.DataFrame,
                              audit: AuditTrail,
                              log: logging.Logger) -> pd.DataFrame:
    """
    Calcula coeficientes médios, dispersão e tipologia XYZ por par.

    Métricas (DOC-PROJ §10.2.2):
        • Coeficientes médios: 4 versões em paralelo (consumo, demanda, env, rec)
        • Dispersão: σ, CV
        • Decomposição regular/extra: % extra, razão extra/regular
        • Atendimento: gap médio, taxa de atendimento
    """
    log.info("Camada 2.2 — Métricas por par (unidade × produto)")

    def calc(g: pd.DataFrame) -> pd.Series:
        consumo_total = g['consumo_total']
        consumo_reg = g['consumo_reg']
        consumo_extra = g['consumo_extra']
        demanda_total = g['demanda_total']

        mu = consumo_total.mean()
        sigma = consumo_total.std(ddof=0)

        return pd.Series({
            # Coeficientes médios (4 métricas-base)
            'consumo_medio_mensal': mu,
            'demanda_media_mensal': demanda_total.mean(),
            'env_medio_mensal': (g['env_reg'] + g['env_extra']).mean(),
            'rec_medio_mensal': (g['rec_reg'] + g['rec_extra']).mean(),
            # Decomposição
            'consumo_reg_medio': consumo_reg.mean(),
            'consumo_extra_medio': consumo_extra.mean(),
            'demanda_reg_media': g['demanda_reg'].mean(),
            'demanda_extra_media': g['demanda_extra'].mean(),
            # Dispersão
            'desvio_padrao': sigma,
            'cv': sigma / mu if mu > 0 else np.nan,
            'meses_com_pedido': int((consumo_total > 0).sum()),
            'meses_com_extra': int((consumo_extra > 0).sum()),
            # Atendimento
            'gap_medio_mensal': g['gap_atendimento'].mean(),
            'taxa_atendimento': (consumo_total.sum() / demanda_total.sum()
                                  if demanda_total.sum() > 0 else np.nan),
            # Pressão dos extras
            'razao_extra_regular': (consumo_extra.sum() / consumo_reg.sum()
                                     if consumo_reg.sum() > 0 else np.nan),
            'pct_extra_no_consumo': (consumo_extra.sum() / consumo_total.sum()
                                      if consumo_total.sum() > 0 else np.nan),
        })

    metricas = (piv.groupby(['unidade', 'produto', 'categoria'])
                   .apply(calc, include_groups=False)
                   .reset_index())

    # Tipologia XYZ (DOC-PROJ §10.2.3)
    def classificar_xyz(cv: float, mu: float) -> str:
        if pd.isna(cv):                       return 'sem_consumo'
        if mu < 1.0:                          return 'baixa_rotatividade'
        if cv < CONFIG.CV_CORTE_REGULAR:      return 'X_regular'
        if cv < CONFIG.CV_CORTE_MODERADO:     return 'Y_moderado'
        return 'Z_erratico'

    metricas['classe_xyz'] = metricas.apply(
        lambda r: classificar_xyz(r['cv'], r['consumo_medio_mensal']), axis=1
    )

    audit.log('2.2_metricas_por_par', len(piv), len(metricas),
              f'{len(metricas)} pares com métricas e classificação XYZ')
    log.info(f"  → distribuição XYZ:\n{metricas['classe_xyz'].value_counts().to_string()}")
    return metricas


# ============================================================================
# CAMADA 3 — POLÍTICA DE COTA MENSAL
# Referência: DOC-PROJ §10.3
# ============================================================================

def camada3_politica_cota(piv: pd.DataFrame,
                            metricas: pd.DataFrame,
                            audit: AuditTrail,
                            log: logging.Logger) -> pd.DataFrame:
    """
    Calcula duas políticas de cota por par (DOC-PROJ §10.3):

        Política CONSERVADORA: cota = média móvel ponderada (sem buffer)
            ŷ_{t+1} = Σ w_i · y_{t-i+1}

        Política COM BUFFER: cota = projeção + k(CV, μ) · σ
            ŷ_{t+1} = MMP + k · σ_y

    A função k(CV, μ) é em escada (ver CONFIG e DOC-PROJ §10.3.2).
    """
    log.info("Camada 3 — Política de cota mensal")

    # Média móvel ponderada por par
    def mmp(g: pd.DataFrame) -> float:
        g_sorted = g.sort_values('mes')
        vals = g_sorted['consumo_total'].values
        if len(vals) == len(CONFIG.PESOS_MMP):
            return float(np.dot(vals, CONFIG.PESOS_MMP))
        return float(vals.mean()) if len(vals) > 0 else 0.0

    proj = (piv.groupby(['unidade', 'produto', 'categoria'])
                .apply(mmp, include_groups=False)
                .reset_index(name='proj_ponderada'))
    politicas = metricas.merge(proj, on=['unidade', 'produto', 'categoria'])

    # Função k(CV, μ) em escada
    def k(row) -> float:
        if pd.isna(row['cv']) or row['consumo_medio_mensal'] < 1.0:
            return CONFIG.K_BAIXA_ROTATIVIDADE
        if row['cv'] < CONFIG.CV_CORTE_REGULAR:
            return CONFIG.K_BAIXA_ROTATIVIDADE
        if row['cv'] < CONFIG.CV_CORTE_MODERADO:
            return CONFIG.K_MODERADO
        if row['cv'] < 1.20:
            return CONFIG.K_ERRATICO_BAIXO
        return CONFIG.K_ERRATICO_ALTO

    politicas['k'] = politicas.apply(k, axis=1)

    # Duas políticas: conservadora e com buffer
    politicas['cota_conservadora'] = politicas['proj_ponderada'].round(2)
    politicas['cota_buffer'] = (politicas['proj_ponderada']
                                  + politicas['k'] * politicas['desvio_padrao']).round(2)

    # Necessidade marginal = cota proposta − consumo regular atual
    politicas['necessidade_marginal_conserv'] = (
        politicas['cota_conservadora'] - politicas['consumo_reg_medio']
    ).round(2)
    politicas['necessidade_marginal_buffer'] = (
        politicas['cota_buffer'] - politicas['consumo_reg_medio']
    ).round(2)

    # Ação sugerida (regra de decisão DOC-PROJ §10.3.4)
    def acao(r) -> str:
        if r['consumo_medio_mensal'] < CONFIG.CONSUMO_MEDIO_MIN:
            return 'baixa_materialidade'
        cota_int = int(np.ceil(r['cota_conservadora']))
        atual_int = int(round(r['consumo_reg_medio']))
        if r['pct_extra_no_consumo'] >= 0.40 and cota_int > atual_int:
            return 'reforcar'
        if cota_int > atual_int + 1: return 'aumentar'
        if cota_int < atual_int - 1: return 'reduzir'
        return 'manter'

    politicas['acao_sugerida'] = politicas.apply(acao, axis=1)

    audit.log('3_politica_cota', len(metricas), len(politicas),
              'Cotas conservadora e com buffer + ação sugerida')
    log.info(f"  → ações: \n{politicas['acao_sugerida'].value_counts().to_string()}")
    return politicas


# ============================================================================
# CAMADA 4 — ANÁLISE DE DOMINÂNCIA ESTOCÁSTICA ENTRE POLÍTICAS
# Referência: DOC-PROJ §10.4
# ============================================================================

def camada4_dominancia_estocastica(piv: pd.DataFrame,
                                     politicas: pd.DataFrame,
                                     audit: AuditTrail,
                                     log: logging.Logger) -> pd.DataFrame:
    """
    Compara políticas de cota via análise de dominância estocástica de 2a ordem
    (SSD), simulando a distribuição de custo total sob cada política.

    Função de custo por par (unidade × produto):
        C(cota) = c_falta · max(0, demanda − cota)
                + c_estoque · max(0, cota − demanda)

    Pesos default: c_falta = 1.0, c_estoque = 0.3 (assumimos que falta custa
    mais que excesso — DOC-PROJ §10.4.3).

    Bootstrap não-paramétrico: reamostra com reposição a série de consumo
    observado de cada par para gerar a CDF empírica do custo.

    Critério SSD: política A domina B em 2a ordem se, para todo z,
        ∫_{-∞}^{z} F_A(t) dt <= ∫_{-∞}^{z} F_B(t) dt
    com desigualdade estrita em algum ponto. Implementado via comparação das
    integrais acumuladas das CDFs amostrais.
    """
    log.info("Camada 4 — Dominância estocástica entre políticas (SSD)")

    rng = np.random.default_rng(SEED)
    C_FALTA = 1.0
    C_ESTOQUE = 0.3

    def custo_simulado(demanda_real: np.ndarray, cota: float) -> np.ndarray:
        falta = np.maximum(0, demanda_real - cota)
        excesso = np.maximum(0, cota - demanda_real)
        return C_FALTA * falta + C_ESTOQUE * excesso

    def bootstrap_custos(demanda_serie: np.ndarray, cota: float, n: int) -> np.ndarray:
        amostras = rng.choice(demanda_serie, size=n, replace=True)
        return custo_simulado(amostras, cota)

    def ssd_domina(custos_a: np.ndarray, custos_b: np.ndarray) -> str:
        """
        Verifica dominância estocástica de 2a ordem entre custos.
        Custos menores são preferíveis. Retorna 'A', 'B' ou 'nenhuma'.
        """
        grid = np.linspace(min(custos_a.min(), custos_b.min()),
                            max(custos_a.max(), custos_b.max()), 200)
        cdf_a = np.array([np.mean(custos_a <= z) for z in grid])
        cdf_b = np.array([np.mean(custos_b <= z) for z in grid])
        # Integrais acumuladas das CDFs (SSD: menor integral é melhor)
        int_a = np.cumsum(cdf_a)
        int_b = np.cumsum(cdf_b)
        if np.all(int_a <= int_b) and np.any(int_a < int_b):
            return 'A'
        if np.all(int_b <= int_a) and np.any(int_b < int_a):
            return 'B'
        return 'nenhuma'

    resultados = []
    # Apenas pares materiais (consumo médio >= 0.5)
    politicas_mat = politicas[politicas['consumo_medio_mensal']
                                >= CONFIG.CONSUMO_MEDIO_MIN].copy()

    for _, row in politicas_mat.iterrows():
        serie = piv[(piv['unidade'] == row['unidade'])
                    & (piv['produto'] == row['produto'])
                    ]['consumo_total'].values
        if len(serie) == 0 or serie.sum() == 0:
            continue

        # Três políticas: atual, conservadora, com buffer
        cota_atual = max(round(row['consumo_reg_medio']), 0)
        cota_conserv = max(np.ceil(row['cota_conservadora']), 0)
        cota_buffer = max(np.ceil(row['cota_buffer']), 0)

        c_atual   = bootstrap_custos(serie, cota_atual,   CONFIG.N_BOOTSTRAP)
        c_conserv = bootstrap_custos(serie, cota_conserv, CONFIG.N_BOOTSTRAP)
        c_buffer  = bootstrap_custos(serie, cota_buffer,  CONFIG.N_BOOTSTRAP)

        # Comparações pareadas
        dom_conserv_vs_atual = ssd_domina(c_conserv, c_atual)
        dom_buffer_vs_atual  = ssd_domina(c_buffer,  c_atual)
        dom_buffer_vs_conserv = ssd_domina(c_buffer, c_conserv)

        # Política preferida = menor custo esperado entre as não-dominadas
        custos_esperados = {
            'atual': c_atual.mean(),
            'conservadora': c_conserv.mean(),
            'com_buffer': c_buffer.mean()
        }
        politica_pref = min(custos_esperados, key=custos_esperados.get)

        resultados.append({
            'unidade': row['unidade'],
            'produto': row['produto'],
            'categoria': row['categoria'],
            'cota_atual': cota_atual,
            'cota_conservadora': cota_conserv,
            'cota_buffer': cota_buffer,
            'custo_esp_atual': c_atual.mean(),
            'custo_esp_conserv': c_conserv.mean(),
            'custo_esp_buffer': c_buffer.mean(),
            'dom_conserv_vs_atual': dom_conserv_vs_atual,
            'dom_buffer_vs_atual': dom_buffer_vs_atual,
            'dom_buffer_vs_conserv': dom_buffer_vs_conserv,
            'politica_preferida': politica_pref,
        })

    dom = pd.DataFrame(resultados)
    audit.log('4_dominancia_estocastica', len(politicas_mat), len(dom),
              f'SSD via bootstrap n={CONFIG.N_BOOTSTRAP}, c_falta={C_FALTA}, c_estoque={C_ESTOQUE}')
    log.info(f"  → política preferida (frequência):\n"
             f"{dom['politica_preferida'].value_counts().to_string()}")
    return dom


# ============================================================================
# ORQUESTRADOR
# ============================================================================

def executar_pipeline(input_path: Path, output_dir: Path,
                      anonimizar: bool = False) -> dict:
    """Executa o pipeline completo end-to-end."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log = configurar_logging(output_dir)
    audit = AuditTrail(output_dir)

    log.info("=" * 70)
    log.info("INÍCIO DO PIPELINE — Eficiência Alocativa sob Incerteza Estocástica")
    log.info(f"Seed: {SEED} | Versão CONFIG: {asdict(CONFIG)}")
    log.info("=" * 70)

    # --- Camada 1 ---
    df = camada1_carregar_e_normalizar(input_path, audit, log)
    df = camada1_anonimizar(df, audit, log, ativo=anonimizar)
    df = camada1_flags_qualidade(df, audit, log)
    df = camada1_consumo_consolidado(df, audit, log)
    df.to_csv(output_dir / '01_dataset_consolidado.csv', index=False, encoding='utf-8')

    # --- Camada 2 ---
    piv = camada2_agregar_mensal(df, audit, log)
    metricas = camada2_metricas_por_par(piv, audit, log)
    metricas.to_csv(output_dir / '02_metricas_por_par.csv', index=False, encoding='utf-8')

    # --- Camada 3 ---
    politicas = camada3_politica_cota(piv, metricas, audit, log)
    politicas.to_csv(output_dir / '03_politicas_cota.csv', index=False, encoding='utf-8')

    # --- Camada 4 ---
    dominancia = camada4_dominancia_estocastica(piv, politicas, audit, log)
    dominancia.to_csv(output_dir / '04_dominancia.csv', index=False, encoding='utf-8')

    audit.save()
    log.info("=" * 70)
    log.info(f"PIPELINE CONCLUÍDO. Saídas em: {output_dir}")
    log.info("=" * 70)

    return {
        'dataset': df, 'painel': piv, 'metricas': metricas,
        'politicas': politicas, 'dominancia': dominancia
    }


def executar_modulo_ml(
    resultado_pipeline: dict,
    data_dir: Path,
    output_dir: Path,
    log: logging.Logger,
) -> dict:
    """
    Executa as Camadas 5 e 6 (módulo ML) sobre o resultado do pipeline clássico.

    Encapsula: ingestão incremental do histórico, backtest walk-forward,
    retreino versionado, detecção de drift, previsão T+1 e fechamento do ciclo.

    Parameters
    ----------
    resultado_pipeline : dict
        Saída de executar_pipeline (chaves: 'painel', 'metricas', 'dominancia').
    data_dir : Path
        Diretório de estado persistente (historico_mensal.csv, model_registry.csv).
    output_dir : Path
        Diretório de saída (05_previsoes.csv, 06_backtest_walkforward.csv).
    log : Logger

    Returns
    -------
    dict : {campeao, metricas_campeao, drift_status, previsoes, dominancia_ml}
    """
    # Importação lazy — módulo ML é opcional
    from ml_forecasting import camada5_prever
    from ml_validation import (
        IngestorIncremental, ModelRegistry, DriftMonitor,
        camada6_retreinar, fechar_ciclo,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    hist_path = data_dir / 'historico_mensal.csv'
    reg_path = data_dir / 'model_registry.csv'

    # Ingestão incremental
    ingestor = IngestorIncremental(hist_path)
    historico = ingestor.append(resultado_pipeline['painel'], log=log)

    metricas_xyz = resultado_pipeline['metricas']
    registry = ModelRegistry(reg_path)
    drift_monitor = DriftMonitor(
        janela=ML_CONFIG.DRIFT_JANELA,
        z_threshold=ML_CONFIG.DRIFT_Z_THRESHOLD,
    )

    # Camada 6 — backtest, retreino, drift
    resultado_c6 = camada6_retreinar(
        historico=historico,
        metricas_xyz=metricas_xyz,
        t_preliminar=ML_CONFIG.T_MIN_PRELIMINAR,
        t_operacional=ML_CONFIG.T_OPERACIONAL,
        ridge_alphas=ML_CONFIG.RIDGE_ALPHAS,
        registry=registry,
        drift_monitor=drift_monitor,
        output_dir=output_dir,
        log=log,
    )

    # Camada 5 — previsão T+1
    previsoes = camada5_prever(
        historico=historico,
        metricas_xyz=metricas_xyz,
        t_preliminar=ML_CONFIG.T_MIN_PRELIMINAR,
        t_operacional=ML_CONFIG.T_OPERACIONAL,
        ridge_alphas=ML_CONFIG.RIDGE_ALPHAS,
        modelo_campeao=resultado_c6['campeao'],
        output_dir=output_dir,
        log=log,
    )

    # Fechamento do ciclo — enriquece análise de dominância com demanda prevista
    dominancia_ml = fechar_ciclo(
        dominancia_df=resultado_pipeline['dominancia'],
        previsoes=previsoes,
        log=log,
    )
    dominancia_ml.to_csv(output_dir / '04_dominancia_ml.csv', index=False, encoding='utf-8')
    log.info('Camadas 5-6 concluídas. Saídas: 05_previsoes.csv, '
             '06_backtest_walkforward.csv, 04_dominancia_ml.csv')

    return {
        'campeao': resultado_c6['campeao'],
        'metricas_campeao': resultado_c6['metricas_campeao'],
        'drift_status': resultado_c6['drift_status'],
        'previsoes': previsoes,
        'dominancia_ml': dominancia_ml,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', required=True, type=Path,
                        help='Caminho do Relatorio_SCM.xls')
    parser.add_argument('--output', required=True, type=Path,
                        help='Diretório de saída')
    parser.add_argument('--anonimizar', action='store_true',
                        help='Remapeia unidades para Unidade A..O (modo publicação)')
    parser.add_argument('--ml', action='store_true',
                        help='Executa Camadas 5-6 (módulo ML): previsão e retreino versionado')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Diretório de estado persistente do ML '
                             '(historico_mensal.csv, model_registry.csv). '
                             'Default: <output>/data/')
    args = parser.parse_args()

    resultado = executar_pipeline(args.input, args.output, anonimizar=args.anonimizar)

    if args.ml:
        log = logging.getLogger(__name__)
        data_dir = args.data_dir or args.output / 'data'
        executar_modulo_ml(resultado, data_dir, args.output, log)


if __name__ == '__main__':
    main()
