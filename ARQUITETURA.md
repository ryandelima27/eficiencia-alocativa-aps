# Arquitetura e Funcionamento do Modelo
## Eficiência Alocativa de Materiais sob Incerteza Estocástica

> **Uso deste documento.** Descreve com precisão o que o sistema faz, como
> cada camada funciona e onde cada parâmetro vive. Serve como contexto para
> colaboradores, para sessões futuras de desenvolvimento assistido e para
> avaliação metodológica independente.

---

## 1. Problema e enquadramento

Uma rede de Atenção Primária à Saúde abastece suas unidades por **cota mensal
regular** complementada por **pedidos extras** quando a cota é insuficiente.
Com 21,1% dos pedidos sendo extras (base jan–abr/2026), há evidência de cotas
mal dimensionadas — algumas unidades pedem demais via extra, outras racionam.

**O objeto do modelo não é prever consumo pontual.** O enquadramento é de
**eficiência alocativa sob incerteza estocástica**: dado que a demanda futura
é desconhecida, qual política de cota produz a melhor distribuição de custo
esperado (falta + excesso)? A resposta é dada por **dominância estocástica de
2ª ordem (SSD)** entre políticas alternativas.

> **Regra de vocabulário.** Média móvel ponderada, função de buffer e bootstrap
> são estatística clássica — não são ML. O módulo ML (Camadas 5-6) usa essa
> designação apenas porque cumpre os requisitos: função-perda minimizada,
> parâmetros aprendidos dos dados e validação out-of-sample. Nada é chamado de
> "preditivo" sem esses três elementos.

---

## 2. Fonte de dados

| Atributo | Descrição |
|---|---|
| **Origem** | Relatório SCM exportado como HTML com extensão `.xls` |
| **Leitura** | `pandas.read_html` — não engine binário (o arquivo é HTML) |
| **Granularidade** | Uma linha por pedido |
| **Colunas** | unidade, data_solic, produto, categoria, qtd_solic, qtd_env, qtd_rec, status, pedido_extra, mes_ref |
| **Volume atual** | 2.603 pedidos · 15 unidades · 125 produtos · jan–abr/2026 (T = 4 meses completos) |
| **Período parcial** | Maio/2026 entra só na auditoria; fica fora das médias |
| **Confidencialidade** | Unidades anonimizadas (A–O) em artefatos publicáveis; arquivo original nunca versionado |

---

## 3. Arquitetura geral — 6 camadas

```
Relatório SCM (.xls)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  CAMADA 1 — Consolidação e auditoria              │  pipeline_eficiencia_alocativa.py
│  Normaliza, detecta erros, constrói qtd_consumo  │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  CAMADA 2 — Caracterização e tipologia XYZ        │  pipeline_eficiencia_alocativa.py
│  Agrega mensalmente, calcula métricas por par     │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  CAMADA 3 — Política de cota mensal               │  pipeline_eficiencia_alocativa.py
│  MMP + buffer k(CV, μ) → cota conserv. e buffer  │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  CAMADA 4 — Dominância estocástica (SSD)          │  pipeline_eficiencia_alocativa.py
│  Bootstrap 5.000 réplicas · compara 3 políticas  │
└───────────────────────┬───────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │  historico_mensal.csv       │  (estado persistente)
          ▼                            │
┌─────────────────────────┐            │
│  CAMADA 5 — Previsão   │            │
│  ml_forecasting.py      │────────────┘
│  Gate · Features · Fit  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  CAMADA 6 — Validação  │
│  ml_validation.py       │
│  Walk-forward · Registry│
│  Drift · fechar_ciclo   │
└─────────────────────────┘
```

---

## 4. Camadas 1–4 — Pipeline clássico

### Camada 1 · Consolidação e auditoria
**Arquivo:** `pipeline_eficiencia_alocativa.py` · funções `camada1_*`

**Variável-alvo `qtd_consumo`** — construída por árvore hierárquica:

```
status = Rejeitado          → 0
qtd_rec > 0                 → min(qtd_rec, qtd_env)
qtd_rec = 0 e qtd_env > 0  → min(qtd_env, qtd_solic)
status em STATUS_PENDENTE   → qtd_solic
caso contrário              → 0
```

Justificativa: ~9% dos pedidos finalizados não têm baixa de recebimento
(subnotificação). A árvore hierárquica rec > env > solic contorna isso sem
superestimar via qtd_solic em pedidos rejeitados.

**Flags de qualidade detectadas:** `flag_env_maior_solic`, `flag_rec_maior_env`,
`flag_fin_sem_rec`. Cada transformação registrada em `log_auditoria.csv`.

---

### Camada 2 · Caracterização e tipologia XYZ
**Arquivo:** `pipeline_eficiencia_alocativa.py` · funções `camada2_*`

Agrega por `(unidade, produto, mes)` e cria painel balanceado com zeros para
meses sem pedido. Calcula por par:

| Métrica | Descrição |
|---|---|
| `consumo_medio_mensal` (μ) | Média do consumo nos T meses |
| `desvio_padrao` (σ) | Desvio-padrão populacional (ddof=0) |
| `cv` | σ / μ (coeficiente de variação) |
| `pct_extra_no_consumo` | Fração do consumo originada de pedidos extras |
| `taxa_atendimento` | consumo_total / demanda_total |

**Tipologia XYZ** (cortes em `CONFIG`):

| Classe | Critério |
|---|---|
| `X_regular` | CV < 0,30 e μ ≥ 1,0 |
| `Y_moderado` | 0,30 ≤ CV < 0,70 e μ ≥ 1,0 |
| `Z_erratico` | CV ≥ 0,70 e μ ≥ 1,0 |
| `baixa_rotatividade` | μ < 1,0 (qualquer CV) |

Itens com μ < 1,0 recebem `baixa_rotatividade` mesmo com CV alto — CV elevado
em baixo volume é artefato matemático, não erraticidade real.

---

### Camada 3 · Política de cota mensal
**Arquivo:** `pipeline_eficiencia_alocativa.py` · função `camada3_politica_cota`

Duas políticas por par:

**Cota conservadora:**
```
ŷ_{t+1} = Σ wᵢ · y_{t−i+1}    pesos = (0,10; 0,20; 0,30; 0,40)
```
Pesos fixos — série curta (T = 4) não sustenta estimação. Se menos de 4 meses
disponíveis, usa média simples.

**Cota com buffer:**
```
cota_buffer = ŷ_{t+1} + k(CV, μ) · σ
```

Função k em escada (parâmetros em `CONFIG`):

| Condição | k |
|---|---|
| μ < 1,0 **ou** CV < 0,30 | 0,50 |
| 0,30 ≤ CV < 0,70 | 1,00 |
| 0,70 ≤ CV < 1,20 | 1,25 |
| CV ≥ 1,20 | 1,50 |

---

### Camada 4 · Dominância estocástica (SSD)
**Arquivo:** `pipeline_eficiencia_alocativa.py` · função `camada4_dominancia_estocastica`

**Função de custo** por par, dada uma política de cota Q e demanda D:
```
C(Q) = c_falta · max(0, D − Q) + c_estoque · max(0, Q − D)
       com  c_falta = 1,0  e  c_estoque = 0,3
```
A razão 1,0 / 0,3 é **arbitrada** (não calibrada com a gestão) — limitação
declarada.

**Bootstrap não-paramétrico:** reamostra com reposição a série histórica de
consumo de cada par (n = 5.000 réplicas) para gerar a CDF empírica do custo
sob cada política.

**Critério SSD:** política A domina B em 2ª ordem se
```
∫_{−∞}^{z} F_A(t) dt ≤ ∫_{−∞}^{z} F_B(t) dt  para todo z
```
com desigualdade estrita em algum ponto. Implementado via integral acumulada
das CDFs amostrais.

Três políticas comparadas em pares: atual × conservadora × com buffer.
Seed fixada: `SEED = 20260501` — reprodutibilidade bit-a-bit.

---

## 5. Camadas 5–6 — Módulo ML

### 5.1 Gate de confiabilidade
**Arquivo:** `ml_forecasting.py` · função `gate_confiabilidade`

| T (meses acumulados) | Confiabilidade | Comportamento |
|---|---|---|
| T < 8 | `insuficiente` | **Modo sombra** — gera previsões, acumula erro, NÃO substitui Camadas 3-4 |
| 8 ≤ T < 18 | `preliminar` | Previsões ativas, alimentam Camada 4, sem sazonalidade |
| T ≥ 18 | `operacional` | Sazonalidade habilitada (ciclos completos disponíveis) |

Cortes em `MLConfig` — não hardcoded.

---

### 5.2 Engenharia de features
**Arquivo:** `ml_forecasting.py` · função `construir_features`

Arquitetura **pooled**: uma única matriz de features para todos os pares
(unidade × produto), não modelos par a par. Isso aumenta o volume efetivo
de dados e permite que o modelo aprenda padrões entre pares.

| Grupo | Features |
|---|---|
| **Lags temporais** | `lag_1` … `lag_4` (consumo em t−1 … t−4); flag de disponibilidade `lag_N_disp` |
| **Estatísticas rolling** | `media_3m`, `std_3m` (janela 3, shift 1 — sem leakage) |
| **Tendência** | `trend_idx` (posição ordinal no histórico do par: 0, 1, 2, …) |
| **Regime** | `regime_X_regular`, `regime_Y_moderado`, `regime_Z_erratico`, `regime_baixa_rotatividade`, `cv_hist`, `consumo_medio_hist` |
| **Dummies categóricas** | One-hot de `categoria` (COPA, ESCRITÓRIO, LIMPEZA) |
| **Dummies de unidade** | One-hot de `unidade` (15 unidades — one-hot preferível a target-encoding com N=15 pequeno) |
| **Sazonalidade** | One-hot de `mes_num` (1–12) — **somente com T ≥ 18** |

Previsão para T+1: acrescenta linha sintética com `consumo_total = NaN` por par;
lags computados automaticamente pelo shift dentro do agrupamento por par.

---

### 5.3 Modelos candidatos
**Arquivo:** `ml_forecasting.py` · classes `ForecastModel` e subclasses

Todos compartilham a interface `fit(X, y) → self` / `predict(X) → ndarray`.

**1 · BaselineMMPModel** — piso obrigatório, sempre presente
- Replica exatamente a MMP da Camada 3 (mesmos pesos)
- Não tem parâmetros a estimar; `fit` é no-op
- Nenhum modelo ML é promovido se não vencer este baseline em MAE

**2 · RidgeModel** — disponível com T ≥ 8
- Ridge com α selecionado por `TimeSeriesSplit` CV (scikit-learn)
- Pipeline: `StandardScaler → RidgeCV`
- Vantagem: estável com features correlacionadas (lags)
- Limitação: linearidade e suposição gaussiana inadequadas para volumes baixos

**3 · LGBMModel** — disponível com T ≥ 8
- LightGBM pooled; objetivo L1 (MAE), robusto a picos de pedidos extras
- `num_leaves=15`, `min_child_samples=10` — conservadores para T curto
- Captura não-linearidades e interações entre features e unidades

**4 · PoissonModel** — disponível com T ≥ 8
- GLM Poisson via statsmodels — tecnicamente correto para dados de contagem
- Garante previsões não-negativas por construção; Var = μ (mais realista que gaussiano para μ < 5)
- Com n_obs < 50, usa features reduzidas (apenas lags + trend) para convergência
- Fallback para média do target se o GLM divergir

---

### 5.4 Seleção e promoção de modelos
**Arquivo:** `ml_validation.py` · classe `ModelRegistry`

```
Modelos disponíveis (condicionais a T e dependências instaladas)
        │
        ▼
WalkForwardValidator: treina [1..t], prevê t+1 → MAE por modelo
        │
        ▼
ModelRegistry.deve_promover(challenger, baseline)?
  Regra 1: challenger.MAE < baseline.MAE         (vencer o piso)
  Regra 2: challenger.MAE < champion.MAE         (vencer o atual)
  Ambas obrigatórias — nenhuma exceção
        │
        ▼
Se promovido → torna-se campeão ativo para camada5_prever
```

O registro vive em `model_registry.csv` com versão (`v0001`, `v0002`, …),
data, métricas e motivo de promoção/rejeição. Imutável — nenhuma linha é
deletada.

---

### 5.5 Persistência incremental
**Arquivo:** `ml_validation.py` · classe `IngestorIncremental`

O pipeline clássico é stateless: cada execução lê o arquivo SCM e produz
resultados do zero. O módulo ML precisa de estado acumulado.

`IngestorIncremental` mantém `historico_mensal.csv`:
- Chave de deduplicação: `(unidade, produto, mes)`
- Em conflito: linha nova sobrescreve (permite recalcular meses parciais)
- `keep='last'` no `drop_duplicates`

T (meses únicos) é calculado sobre este arquivo, não sobre o SCM atual.

---

### 5.6 Detecção de drift
**Arquivo:** `ml_validation.py` · classe `DriftMonitor`

z-score do MAE médio dos últimos `DRIFT_JANELA` meses contra o histórico
completo de erros:

```
z = (media_MAE_recente − media_MAE_historica) / std_MAE_historica
flag recalibrar = True  quando  |z| > DRIFT_Z_THRESHOLD (default 2,0)
```

Limitação declarada: z-score simples detecta apenas mudanças abruptas.
Page-Hinkley (mais sensível a mudanças graduais) está documentado como
upgrade futuro para T ≥ 18.

---

### 5.7 Fechamento do ciclo
**Arquivo:** `ml_validation.py` · função `fechar_ciclo`

Quando `confiabilidade ≥ 'preliminar'`, a previsão da Camada 5 substitui a
demanda empírica na entrada da análise SSD (Camada 4). O resultado fica em
`04_dominancia_ml.csv` com a coluna `fonte_demanda` indicando `'ml_prevista'`
ou `'empirica'` por par.

Com `confiabilidade = 'insuficiente'` (modo sombra), `fonte_demanda = 'empirica'`
para todos os pares — as Camadas 3-4 são preservadas intactas.

---

## 6. Parâmetros centralizados

Todos os parâmetros em dois dataclasses imutáveis (`frozen=True`) no topo de
`pipeline_eficiencia_alocativa.py`. Nenhum magic number espalhado pelo código.

**`Config`** — pipeline clássico:

| Parâmetro | Valor | Significado |
|---|---|---|
| `PESOS_MMP` | (0,10; 0,20; 0,30; 0,40) | Pesos da MMP (antigo → recente) |
| `K_BAIXA_ROTATIVIDADE` | 0,50 | Buffer para μ < 1 ou CV < 0,30 |
| `K_MODERADO` | 1,00 | Buffer para 0,30 ≤ CV < 0,70 |
| `K_ERRATICO_BAIXO` | 1,25 | Buffer para 0,70 ≤ CV < 1,20 |
| `K_ERRATICO_ALTO` | 1,50 | Buffer para CV ≥ 1,20 |
| `N_BOOTSTRAP` | 5.000 | Réplicas para CDF do custo (SSD) |
| `CONSUMO_MEDIO_MIN` | 0,50 | Limiar de materialidade |
| `SEED` | 20260501 | Reprodutibilidade bit-a-bit |

**`MLConfig`** — módulo ML:

| Parâmetro | Valor | Significado |
|---|---|---|
| `T_MIN_PRELIMINAR` | 8 | Meses para ativar ML |
| `T_OPERACIONAL` | 18 | Meses para sazonalidade e modo operacional |
| `WALK_FORWARD_MIN_TREINO` | 3 | Folds mínimos de treino no backtest |
| `RIDGE_ALPHAS` | (0,01 … 100) | Candidatos de α para CV do Ridge |
| `DRIFT_JANELA` | 3 | Janela de meses recentes para z-score |
| `DRIFT_Z_THRESHOLD` | 2,0 | Limite de z para flag de recalibração |

---

## 7. Fluxo de execução

```bash
# Pipeline clássico (Camadas 1–4)
python pipeline_eficiencia_alocativa.py \
    --input  data/Relatorio_SCM.xls \
    --output output/

# Pipeline completo com ML (Camadas 1–6)
python pipeline_eficiencia_alocativa.py \
    --input    data/Relatorio_SCM.xls \
    --output   output/ \
    --ml \
    --data-dir data/
```

Saídas:

| Arquivo | Camada | Conteúdo |
|---|---|---|
| `01_dataset_consolidado.csv` | 1 | Dataset limpo com qtd_consumo e flags |
| `02_metricas_por_par.csv` | 2 | Médias, CV, tipologia XYZ por par |
| `03_politicas_cota.csv` | 3 | Cotas conservadora e buffer + ação sugerida |
| `04_dominancia.csv` | 4 | Resultados SSD entre as 3 políticas |
| `04_dominancia_ml.csv` | 4+5 | SSD com demanda ML (quando confiável) |
| `05_previsoes.csv` | 5 | Previsões T+1 com confiabilidade e intervalo 90% |
| `06_backtest_walkforward.csv` | 6 | Erros por fold, modelo e par |
| `data/historico_mensal.csv` | 6 | Painel histórico acumulado (stateful) |
| `data/model_registry.csv` | 6 | Registro versionado de modelos e promoções |
| `log_auditoria.csv` | 1–4 | Trilha de transformações com contagens |
| `pipeline.log` | todos | Log de execução (console + arquivo) |

---

## 8. Limitações declaradas

O sistema registra abertamente onde seu rigor é inferior ao desejável:

| Limitação | Impacto | Onde |
|---|---|---|
| T = 4 meses (série curta) | Pesos MMP fixos; ML em modo sombra | Camadas 3 e 5 |
| Razão de custos falta/excesso = 1,0/0,3 arbitrada | Sensibilidade da SSD não calibrada | Camada 4 |
| Função k em escada (descontinuidade) | Gradiente artificial nos cortes de CV | Camada 3 |
| Pares tratados como independentes | Não captura dependência cruzada entre produtos | Camadas 3-4 |
| Ausência de variável de estoque | Não distingue falta por cota baixa de falta por estoque acumulado | Camada 1 |
| Heterogeneidade entre unidades não controlada | Unidades grandes/pequenas tratadas no mesmo pooled | Camadas 2-5 |
| Intervalo ML assume resíduos i.i.d. | Subestima incerteza em séries com autocorrelação | Camada 5 |
| Metricas XYZ não recalculadas por fold | Leve overfitting nos walk-forward de regime | Camada 6 |
| Drift detectado por z-score simples | Insensível a mudanças graduais de regime | Camada 6 |

---

## 9. Estrutura de arquivos

```
.
├── pipeline_eficiencia_alocativa.py  # Camadas 1-4 + orquestrador ML
├── ml_forecasting.py                 # Camada 5 — previsão com gate
├── ml_validation.py                  # Camada 6 — validação e retreino
├── dashboard.html                    # Dashboard de KPIs (autocontido)
├── CLAUDE.md                         # Contexto e roadmap (uso interno)
├── README.md                         # Documentação pública do repositório
├── docs/
│   └── Projeto_Eficiencia_Alocativa.docx
├── output/
│   └── Modelagem_Consumo_USFs.xlsx   # Relatório operacional (7 abas)
└── data/                             # Estado persistente do ML (não versionado)
    ├── historico_mensal.csv
    └── model_registry.csv
```

---

*Versão: 2.0.0 · Maio/2026 · Ryan Gabriel de Lima Nascimento*
*Licença do código: MIT · Metodologia citável com atribuição*
