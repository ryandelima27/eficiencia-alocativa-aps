# Modelagem da Eficiência Alocativa de Materiais sob Incerteza Estocástica

Arcabouço analítico replicável para diagnóstico de eficiência alocativa e
redesenho de cotas de materiais de apoio em uma rede de Atenção Primária à
Saúde (APS), sob incerteza estocástica.

> **Nota de confidencialidade.** Este projeto foi desenvolvido a partir de
> dados operacionais reais de uma rede municipal de APS. Todas as
> identificações institucionais e de unidades foram suprimidas: as 15
> unidades destinatárias aparecem como *Unidade A* a *Unidade O*. As métricas,
> taxas e estatísticas agregadas são preservadas integralmente. O mapa de
> reversão não é distribuído.

---

## O problema

Uma rede de APS abastece suas unidades por **cota mensal regular** acrescida de
**pedidos extras** quando a cota é insuficiente. A alta participação dos extras
(16,5% de 2.603 pedidos no período analisado) e a heterogeneidade de consumo
entre unidades sugerem que as cotas estão mal dimensionadas — algumas unidades
pedem demais via extra, outras racionam. Este projeto quantifica essa
ineficiência e propõe cotas redesenhadas com base em evidência.

## Abordagem

A modelagem evita deliberadamente vender-se como "preditiva" num cenário em que
os dados (4 meses) não sustentam previsão paramétrica confiável. O enquadramento
é de **eficiência alocativa sob incerteza estocástica**: o objeto não é prever o
consumo pontual, mas avaliar *qual política de cota tem a melhor distribuição de
custo* (falta + excesso) sob a demanda observada, usando **dominância
estocástica de segunda ordem (SSD)**.

### Arquitetura em 4 camadas

| Camada | Função | Entrega |
|--------|--------|---------|
| 1 | Consolidação e auditoria do dado | `01_dataset_consolidado.csv` + flags de erro |
| 2 | Caracterização descritiva e tipologia XYZ | `02_metricas_por_par.csv` |
| 3 | Política de cota (conservadora e com buffer) | `03_politicas_cota.csv` |
| 4 | Dominância estocástica entre políticas | `04_dominancia.csv` |

## Achado principal

Sobre a base de 4 meses, **94% das políticas de cota vigentes são dominadas
estocasticamente em 2ª ordem** por pelo menos uma das alternativas propostas
(cota conservadora ou cota com buffer), considerando uma função de custo que
penaliza falta com peso superior ao excesso de estoque (razão 3,33:1).

## Conteúdo do repositório

```
.
├── README.md                              # este arquivo
├── docs/
│   └── Projeto_Eficiencia_Alocativa.docx  # documento técnico completo (PMI-like)
├── src/
│   └── pipeline_eficiencia_alocativa.py   # pipeline replicável (4 camadas)
└── output/
    └── Modelagem_Consumo_USFs.xlsx        # relatório operacional (7 abas)
```

## Como executar

```bash
pip install pandas numpy openpyxl scipy

# Uso interno (dados nominais)
python src/pipeline_eficiencia_alocativa.py \
    --input  caminho/Relatorio_SCM.xls \
    --output output/

# Modo publicação (anonimiza unidades para A..O)
python src/pipeline_eficiencia_alocativa.py \
    --input  caminho/Relatorio_SCM.xls \
    --output output/ \
    --anonimizar
```

## Garantias de replicabilidade

- **Reprodutibilidade:** seed fixada (`SEED = 20260501`); duas execuções com o
  mesmo input produzem outputs idênticos.
- **Parâmetros versionados:** todos os parâmetros do modelo concentrados no
  objeto imutável `CONFIG` no topo do script.
- **Trilha de auditoria:** cada transformação é registrada em
  `log_auditoria.csv` com contagem de linhas antes/depois e justificativa.
- **Especificação formal:** o documento técnico (Anexo A) traz todas as
  fórmulas, suficientes para reimplementação independente.

## Limitações declaradas

O projeto declara abertamente onde seu rigor é inferior ao desejável: painel
curto (T = 4), pesos da média móvel fixos por escolha (não estimados), função
de buffer descontínua, razão de custos arbitrada, pares tratados como
independentes, ausência de variável de estoque, heterogeneidade entre unidades
não-controlada. Detalhes na Seção 14 do documento técnico.

## Roadmap

Acumular base ≥ 8 meses; calibrar custos com a gestão; análise de sensibilidade;
suavizar a função de buffer; modelar dependência cruzada entre produtos;
incorporar variáveis de heterogeneidade; integrar leitura programática do
sistema-fonte.

## Stack

Python · pandas · numpy · openpyxl · scipy · bootstrap não-paramétrico ·
dominância estocástica de 2ª ordem (SSD)

## Licença

Código sob MIT. Documento e metodologia podem ser citados com atribuição.

## Autor

**Ryan Gabriel de Lima Nascimento** — Ciências Atuariais (FEA-USP)
Foco em risco, modelagem estocástica e inteligência de dados.
