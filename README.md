# Eficiência Alocativa de Estoques sob Incerteza Estocástica

Arcabouço analítico replicável para diagnóstico de eficiência alocativa e
redesenho de políticas de cota em redes com reposição periódica, **pedidos
regulares** e **requisições complementares** quando a cota se mostra
insuficiente.

> **Dataset sintético.** O repositório inclui dataset sintético calibrado
> para preservar as propriedades estatísticas relevantes ao problema
> (proporção de extras ~21 %, estrutura de variabilidade por categoria,
> tipologia XYZ 40/30/30 %) sem correspondência com nenhuma operação real.
> O arcabouço é replicável a qualquer contexto de alocação com função de
> custo assimétrica.

---

## O problema

Organizações que usam **cotas fixas calibradas pela média histórica simples**
produzem sistematicamente dois erros simultâneos:

- **Falta** onde a demanda é volátil — unidades com padrão errático pedindo
  via extras o que a cota não cobre.
- **Excesso** onde a demanda é previsível — capital imobilizado em itens de
  baixo consumo.

A alta participação de requisições complementares (extras) é um sinal
observável de que as cotas estão mal dimensionadas. Este projeto quantifica
essa ineficiência e propõe cotas redesenhadas com base em evidência
estocástica — não em médias pontuais.

## Abordagem

O enquadramento é de **eficiência alocativa sob incerteza estocástica**: o
objeto não é prever o consumo pontual, mas avaliar *qual política de cota tem
a melhor distribuição de custo* (falta + excesso) sob a demanda observada,
usando **dominância estocástica de segunda ordem (SSD)**.

O modelo deliberadamente não se apresenta como preditivo em cenários com
série curta (T < 8 meses). A maturidade analítica cresce conforme o histórico
acumulado.

### Arquitetura em 4+2 camadas

| Camada | Função | Entrega |
|--------|--------|---------|
| 1 | Consolidação e auditoria do dado | `01_dataset_consolidado.csv` |
| 2 | Caracterização descritiva e tipologia XYZ | `02_metricas_por_par.csv` |
| 3 | Política de cota (conservadora e com buffer) | `03_politicas_cota.csv` |
| 4 | Dominância estocástica entre políticas | `04_dominancia.csv` |
| 5 | Previsão de demanda com gate de confiabilidade | `05_previsoes.csv` |
| 6 | Backtesting walk-forward, retreino e drift | `06_backtest_walkforward.csv` |

## Achado principal (dataset de demonstração)

Sobre o dataset sintético de demonstração, **mais de 9 em cada 10 pares têm
sua cota vigente dominada estocasticamente em 2ª ordem** por pelo menos uma
das alternativas propostas (cota conservadora ou cota com buffer).

> **Condicional importante.** Este resultado é sensível à razão de custo
> adotada (falta/excesso = 3,33:1). Razões diferentes produzem resultados
> diferentes; a seção de Limitações detalha como calibrar com a operação real.

## Dataset sintético

O arquivo `data/dataset_sintetico.csv` foi gerado por `generate_synthetic_data.py`
com seed fixada (`SEED = 20260501`) e as seguintes propriedades:

| Propriedade | Valor |
|---|---|
| Unidades | 15 (Unidade_01 … Unidade_15) |
| Produtos | 110 em 3 categorias genéricas |
| Pares materiais (μ ≥ 0,5/mês) | ~600 |
| Proporção de extras | ~21 % do consumo total |
| Tipologia XYZ | X ≈ 35 % · Y ≈ 33 % · Z ≈ 32 % |
| Período completo | Jan–Abr/2026 (T = 4 meses) |
| Período parcial | Mai/2026 (só auditoria) |

## Como executar

```bash
pip install pandas numpy openpyxl scipy scikit-learn lightgbm statsmodels

# Dataset sintético (incluído no repo)
python generate_synthetic_data.py          # gera data/dataset_sintetico.csv
python pipeline_eficiencia_alocativa.py \
    --input  data/dataset_sintetico.csv \
    --output output/

# Com módulo ML (Camadas 5-6)
python pipeline_eficiencia_alocativa.py \
    --input    data/dataset_sintetico.csv \
    --output   output/ \
    --ml \
    --data-dir data/

# Dados próprios em formato CSV (colunas padronizadas)
python pipeline_eficiencia_alocativa.py \
    --input  /caminho/seu_arquivo.csv \
    --output output/
```

## Garantias de replicabilidade

- **Reprodutibilidade:** seed fixada (`SEED = 20260501`); duas execuções com
  o mesmo input produzem outputs idênticos.
- **Parâmetros versionados:** todos os parâmetros do modelo concentrados nos
  objetos imutáveis `CONFIG` e `MLConfig` no topo do script.
- **Trilha de auditoria:** cada transformação registrada em `log_auditoria.csv`
  com contagem de linhas antes/depois e justificativa.
- **Especificação formal:** `ARQUITETURA.md` detalha todas as fórmulas,
  suficientes para reimplementação independente.

## Limitações declaradas

| Limitação | Impacto |
|---|---|
| T = 4 meses (série curta) | Pesos MMP fixos; ML em modo sombra |
| Razão de custos falta/excesso = 1,0/0,3 arbitrada | Sensibilidade da SSD não calibrada com a operação |
| Função k em escada (descontinuidade) | Gradiente artificial nos cortes de CV |
| Pares tratados como independentes | Não captura dependência cruzada entre itens |
| Heterogeneidade entre unidades não controlada | Unidades grandes/pequenas no mesmo pooled |
| Intervalo ML assume resíduos i.i.d. | Subestima incerteza em séries com autocorrelação |

## Roadmap

- Acumular base ≥ 8 meses para ativar modelos ML (Ridge, LightGBM, Poisson GLM)
- Calibrar razão de custos com a equipe de gestão
- Análise de sensibilidade sobre os parâmetros da função k
- Suavizar a função de buffer (eliminar descontinuidade)
- Modelar dependência cruzada entre itens correlacionados
- Incorporar variável de estoque disponível

## Stack

Python · pandas · numpy · openpyxl · scipy · scikit-learn · lightgbm ·
statsmodels · bootstrap não-paramétrico · dominância estocástica de 2ª ordem (SSD)

## Licença

Código sob MIT. Metodologia pode ser citada com atribuição.

## Autor

**Ryan Gabriel de Lima Nascimento** — Ciências Atuariais (FEA-USP)  
Foco em risco, modelagem estocástica e inteligência de dados.
