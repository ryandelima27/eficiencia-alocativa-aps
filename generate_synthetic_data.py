"""
================================================================================
GERADOR DE DATASET SINTÉTICO — Alocação Estocástica de Estoques
================================================================================

Gera dataset sintético calibrado sem correspondência com qualquer operação real.

ALVOS (tolerância ±5 p.p.)
--------------------------
    % extras   : ~21 % do consumo (meses completos)
    Unidades   : 15   (Unidade_01 … Unidade_15)
    Produtos   : 110  em 3 categorias genéricas
    Pares mat. : 580–640  (μ ≥ 0,5 /mês, meses completos)
    XYZ        : X ≈ 40 %  ·  Y ≈ 30 %  ·  Z ≈ 30 %
    Período    : jan–abr/2026 (T=4) + mai/2026 parcial

DECISÕES DE CALIBRAÇÃO
-----------------------
    1. Perfil XYZ atribuído por PAR (não por produto) com proporções fixas
       X:Y:Z = 240:180:180 de 600 pares ativos → distribuição garantida.
       Com T=4 e heterogeneidade de unidades, perfil por produto não garante
       os alvos: produtos X em unidades pequenas produzem CV empírico ≥ 0,30.

    2. Geração de demanda mensal ajustada ao perfil atribuído:
       - X (μ ≥ 12): Poisson(lam_X) onde lam_X = max(lam_par, 12)
                      → CV teórico ≤ 0,29 para todos os 4 meses ✓
       - Y (μ ∈ [3,8]): Neg-Binomial(r=3, p calibrado) com lam_Y ∈ [3,8]
                         → CV empírico típico 0,35–0,65 ✓
       - Z (zeros forçados): 2 ou 3 zeros nos 4 meses + spike nos demais
                             → CV empírico sempre ≥ 0,87 ✓

    3. Quota regular = floor(E_ativo × 0,79) onde:
       - E_ativo_X/Y = lam mensal (sem zeros)
       - E_ativo_Z   = avg_spike × lam_Z (E[demanda | período ativo])
       → E[extras] ≈ 21 % do consumo em todos os perfis ✓

    4. Subnotificação: ~9 % dos pedidos Finalizados têm qtd_rec = 0.

    5. Jan/2026 frac=0,65 (rampa); Mai/2026 frac=0,35 (parcial).

SEED = 20260501
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

SEED = 20260501
RNG  = np.random.default_rng(SEED)

N_UNITS    = 15
N_PRODUTOS = 110
N_PARES    = 600   # pares ativos totais

# Calibração XYZ via Monte Carlo (T=4, frac_Jan=0.65, Poisson):
#   P(X | lam=40) = 91%   P(Y | lam=40) = 9%   P(Z | lam=40) = 0%
#   P(X | lam=5)  = 26%   P(Y | lam=5)  = 68%  P(Z | lam=5)  = 6%
#   Z-perfil (zeros forçados)                   P(Z) = 100%
#
# Sistema n_X + n_Y + n_Z = 600, alvos X=240, Y=180, Z=180:
#   n_X=197, n_Y=238, n_Z=165
#   X-class = 197×0.91 + 238×0.26 = 179+62 = 241 ≈ 40% ✓
#   Y-class = 197×0.09 + 238×0.68 = 18+162 = 180 = 30% ✓
#   Z-class = 165     + 238×0.06  = 165+14  = 179 ≈ 30% ✓
XYZ_COUNTS = {'X': 197, 'Y': 238, 'Z': 165}

# Faixas de lam para cada perfil
LAM_X_MIN = 40.0   # P(CV<0.30 com T=4, frac_Jan=0.65) ≈ 91%
LAM_Y_LO  =  4.0   # Poisson (não NegBin): cv_teor=1/√lam ∈ [0.33,0.50] → Y ✓
LAM_Y_HI  =  6.0
LAM_Z_LO  =  3.0
LAM_Z_HI  =  9.0

# Parâmetros Z
Z_P_ZERO    = 0.55          # P(forçar zero) — distribui entre n_zero=2 e n_zero=3
Z_SPIKE_LO  = 3.0
Z_SPIKE_HI  = 5.0
Z_AVG_SPIKE = (Z_SPIKE_LO + Z_SPIKE_HI) / 2.0  # 4.0

QUOTA_FRAC = 0.79

MESES = [
    ('2026-01', date(2026, 1,  3), date(2026, 1, 31), 0.65),
    ('2026-02', date(2026, 2,  1), date(2026, 2, 28), 1.00),
    ('2026-03', date(2026, 3,  1), date(2026, 3, 31), 1.00),
    ('2026-04', date(2026, 4,  1), date(2026, 4, 30), 1.00),
    ('2026-05', date(2026, 5,  2), date(2026, 5, 12), 0.35),
]
MESES_COMPLETOS = [m[0] for m in MESES[:4]]
FRACS_COMPLETOS = [m[3] for m in MESES[:4]]

MES_REF = {
    '2026-01': 'Janeiro/2026',   '2026-02': 'Fevereiro/2026',
    '2026-03': 'Março/2026',     '2026-04': 'Abril/2026',
    '2026-05': 'Maio/2026',
}

STATUS_POOL = (
    ['Finalizado'] * 7 + ['Recebimento Confirmado'] * 2 +
    ['Enviada'] * 2 + ['Pendente', 'Aprovado', 'Aguardando Gerente']
)

CATEGORIAS = ['CATEGORIA_A', 'CATEGORIA_B', 'CATEGORIA_C']


# ── Estrutura de produtos e unidades ─────────────────────────────────────────

def _build_produtos() -> pd.DataFrame:
    """110 produtos em 3 categorias genéricas."""
    n = [40, 45, 25]
    rows = []
    idx  = 1
    for cat, qtd in zip(CATEGORIAS, n):
        for _ in range(qtd):
            rows.append({'produto': f'Produto_{idx:03d}', 'categoria': cat,
                         'lam_ref': float(RNG.uniform(3, 20))})
            idx += 1
    return pd.DataFrame(rows)


def _build_unidades() -> pd.DataFrame:
    """15 unidades com fatores de tamanho em ordem decrescente."""
    fatores = np.sort(RNG.uniform(0.35, 2.6, N_UNITS))[::-1]
    return pd.DataFrame({
        'unidade': [f'Unidade_{i:02d}' for i in range(1, N_UNITS + 1)],
        'fator':   fatores,
    })


# ── Série de demanda para 4 meses completos ───────────────────────────────────

def _serie_4meses(lam_ref: float, fator: float,
                  perfil: str, fracs: list[float]) -> list[int]:
    """
    Gera a série de demanda para os 4 meses completos.

    O perfil é atribuído por par para garantir as proporções XYZ alvo.
    O lam_ref × fator é a base, ajustada ao piso de cada perfil.
    """
    if perfil == 'X':
        lam_x = max(lam_ref * fator, LAM_X_MIN)
        return [int(RNG.poisson(max(lam_x * f, 0.5))) for f in fracs]

    elif perfil == 'Y':
        # Poisson com lam_y ∈ [LAM_Y_LO, LAM_Y_HI] → CV teórico 0,41–0,50 → Y ✓
        # NegBin(r=3) foi descartado: CV mínimo teórico = √(1/r)=0.577 (Z range).
        lam_y = float(np.clip(lam_ref * fator, LAM_Y_LO, LAM_Y_HI))
        return [int(RNG.poisson(max(lam_y * f, 0.5))) for f in fracs]

    else:  # Z — zeros forçados para garantir CV ≥ 0.87
        lam_z  = float(np.clip(lam_ref * fator, LAM_Z_LO, LAM_Z_HI))
        n_zero = int(RNG.choice([2, 3]))           # 2 ou 3 zeros nos 4 meses
        zero_idx = set(RNG.choice(4, size=n_zero, replace=False).tolist())
        series = []
        for i, f in enumerate(fracs):
            if i in zero_idx:
                series.append(0)
            else:
                spike = float(RNG.uniform(Z_SPIKE_LO, Z_SPIKE_HI))
                d = int(RNG.poisson(max(lam_z * spike * f, 0.5)))
                series.append(max(d, 2))   # ≥2 garante média ≥ 0,5 mesmo com 3 zeros
        return series


def _quota(lam_ref: float, fator: float, perfil: str) -> int:
    """
    quota = floor(E[demanda_ativa] × QUOTA_FRAC).

    Para X/Y: E[D] = lam (todos meses ativos).
    Para Z:   E[D | ativo] = avg_spike × lam_Z (quota por período NÃO-zero),
              garantindo que extras ≈ 21 % do consumo Z também.
    """
    if perfil == 'X':
        e_dem = max(lam_ref * fator, LAM_X_MIN)
    elif perfil == 'Y':
        e_dem = float(np.clip(lam_ref * fator, LAM_Y_LO, LAM_Y_HI))  # Poisson: E=lam
    else:
        lam_z = float(np.clip(lam_ref * fator, LAM_Z_LO, LAM_Z_HI))
        e_dem = Z_AVG_SPIKE * lam_z
    return max(int(np.floor(e_dem * QUOTA_FRAC)), 0)


# ── Construção de ordens ──────────────────────────────────────────────────────

def _make_row(unidade: str, produto: str, cat: str,
              qtd: int, extra: bool, d0: date, d1: date, mes: str) -> dict:
    status  = str(RNG.choice(STATUS_POOL))
    qtd_env = qtd
    if status in ('Finalizado', 'Recebimento Confirmado'):
        qtd_rec = 0 if RNG.random() < 0.09 else qtd_env
    elif status == 'Enviada':
        qtd_rec = 0
    else:
        qtd_env, qtd_rec = 0, 0
    n_dias = (d1 - d0).days + 1
    data_s = (d0 + timedelta(days=int(RNG.integers(0, n_dias)))).strftime('%d/%m/%Y')
    return {
        'unidade': unidade, 'data_solic': data_s,
        'produto': produto, 'categoria': cat,
        'qtd_solic': qtd, 'qtd_env': qtd_env, 'qtd_rec': qtd_rec,
        'status': status,
        'pedido_extra': 'Sim' if extra else 'Não',
        'mes_ref': MES_REF[mes],
    }


def _ordens_mes(unidade: str, produto: str, cat: str,
                demanda: int, quota: int,
                d0: date, d1: date, mes: str) -> list[dict]:
    if demanda == 0:
        return []
    rows = []
    qtd_reg   = min(quota, demanda) if quota > 0 else demanda
    qtd_extra = max(demanda - quota, 0)
    if qtd_reg   > 0: rows.append(_make_row(unidade, produto, cat, qtd_reg,   False, d0, d1, mes))
    if qtd_extra > 0: rows.append(_make_row(unidade, produto, cat, qtd_extra, True,  d0, d1, mes))
    return rows


# ── Geração principal ─────────────────────────────────────────────────────────

def gerar_dataset(output_path: Path) -> pd.DataFrame:
    """Gera o dataset sintético com perfis XYZ controlados e persiste em CSV."""
    produtos  = _build_produtos()
    unidades  = _build_unidades()

    # Selecionar N_PARES pares aleatórios sem reposição
    pares_possiveis = [(u, p) for u in unidades['unidade']
                               for p in produtos['produto']]
    idx_sel = RNG.choice(len(pares_possiveis), size=N_PARES, replace=False)
    pares_sel = [pares_possiveis[i] for i in idx_sel]

    # Atribuir perfil XYZ (240 X, 180 Y, 180 Z) em ordem aleatória
    perfis = (['X'] * XYZ_COUNTS['X'] +
               ['Y'] * XYZ_COUNTS['Y'] +
               ['Z'] * XYZ_COUNTS['Z'])
    RNG.shuffle(perfis)
    perfil_par = {par: perf for par, perf in zip(pares_sel, perfis)}

    # Lookup rápido para lam_ref e categoria
    prod_info = produtos.set_index('produto')[['lam_ref', 'categoria']].to_dict('index')
    unit_fator = unidades.set_index('unidade')['fator'].to_dict()

    fracs_completos = [m[3] for m in MESES[:4]]
    all_rows: list[dict] = []

    for (uname, pname), perfil in perfil_par.items():
        lam_ref = prod_info[pname]['lam_ref']
        cat     = prod_info[pname]['categoria']
        fator   = unit_fator[uname]

        serie   = _serie_4meses(lam_ref, fator, perfil, fracs_completos)
        q       = _quota(lam_ref, fator, perfil)

        # Meses completos
        for (mes, d0, d1, _), demanda in zip(MESES[:4], serie):
            all_rows.extend(_ordens_mes(uname, pname, cat, demanda, q, d0, d1, mes))

        # Mês parcial (mai): gera com frac=0,35, mesmo perfil
        _, d0p, d1p, fracp = MESES[4]
        d_parcial = _serie_4meses(lam_ref, fator, perfil, [fracp])[0]
        all_rows.extend(_ordens_mes(uname, pname, cat, d_parcial, q, d0p, d1p, '2026-05'))

    df = pd.DataFrame(all_rows)
    df['data_solic'] = pd.to_datetime(df['data_solic'], format='%d/%m/%Y')
    df = df.sort_values(['data_solic', 'unidade', 'produto']).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    return df


# ── Verificação de calibração ─────────────────────────────────────────────────

def verificar_calibracao(df: pd.DataFrame) -> None:
    df_c = df[df['data_solic'].dt.to_period('M').astype(str).isin(MESES_COMPLETOS)].copy()

    total  = df_c['qtd_solic'].sum()
    extras = df_c.loc[df_c['pedido_extra'] == 'Sim', 'qtd_solic'].sum()
    pct_ex = extras / total * 100 if total else 0

    agg = (df_c.assign(mes=df_c['data_solic'].dt.to_period('M').astype(str))
               .groupby(['unidade', 'produto', 'mes'])['qtd_solic']
               .sum().reset_index())
    pares  = agg[['unidade', 'produto']].drop_duplicates()
    grid   = pares.merge(pd.DataFrame({'mes': MESES_COMPLETOS}), how='cross')
    painel = grid.merge(agg, on=['unidade', 'produto', 'mes'], how='left').fillna(0)

    st = (painel.groupby(['unidade', 'produto'])['qtd_solic']
                .agg(['mean', 'std']).reset_index())
    st['cv'] = (st['std'] / st['mean'].replace(0, np.nan)).fillna(0)

    def xyz(r):
        if r['mean'] < 0.5:  return 'inativo'
        if r['mean'] < 1.0:  return 'baixa_rot'
        if r['cv']   < 0.30: return 'X'
        if r['cv']   < 0.70: return 'Y'
        return 'Z'
    st['xyz'] = st.apply(xyz, axis=1)

    mat  = st[st['mean'] >= 0.5]
    dist = mat['xyz'].value_counts(normalize=True) * 100

    print('\n' + '='*58)
    print('CALIBRAÇÃO DO DATASET SINTÉTICO')
    print('='*58)
    print(f'Pedidos totais  : {len(df):,}')
    print(f'Unidades        : {df["unidade"].nunique()}')
    print(f'Produtos        : {df["produto"].nunique()}')
    print(f'% extras        : {pct_ex:.1f}%    [alvo: ~21%]')
    print(f'Pares materiais : {len(mat)}     [alvo: 580–640]')
    print('Tipologia XYZ:')
    for cls, alvo in [('X', '40%'), ('Y', '30%'), ('Z', '30%')]:
        within = abs(dist.get(cls, 0) - int(alvo[:-1])) <= 5
        print(f'  {cls}: {dist.get(cls, 0):5.1f}%   [alvo: {alvo}]  {"✓" if within else "✗"}')
    if dist.get('baixa_rot', 0):
        print(f'  baixa_rot: {dist.get("baixa_rot", 0):.1f}%')

    ok = (abs(pct_ex - 21) <= 5 and
          580 <= len(mat) <= 640 and
          abs(dist.get('X', 0) - 40) <= 5 and
          abs(dist.get('Z', 0) - 30) <= 5)
    print(f'\nStatus          : {"✓  DENTRO DO TOLERADO" if ok else "✗  AJUSTE NECESSÁRIO"}')
    print('='*58 + '\n')


if __name__ == '__main__':
    out = Path('data/dataset_sintetico.csv')
    print(f'Gerando → {out}  (seed={SEED})')
    df = gerar_dataset(out)
    verificar_calibracao(df)
    print(f'Salvo: {len(df):,} linhas')
