# Eficiência Alocativa de Estoques sob Incerteza Estocástica

Arcabouço analítico replicável para diagnóstico de eficiência alocativa e redesenho de políticas de cota em redes com reposição periódica, pedidos regulares e requisições complementares quando a cota se mostra insuficiente.


Motivação intelectual
Organizações que calibram cotas pela média histórica simples cometem um erro de enquadramento antes mesmo de calcular qualquer número: tratam como problema de previsão pontual o que é, na essência, um problema de decisão sob incerteza. A média minimiza o erro quadrático médio — não o custo esperado assimétrico de falta versus excesso. Onde a penalidade de ruptura excede sistematicamente o custo de carregamento, a política ótima não é a que acerta o centro da distribuição; é a que posiciona o estoque no quantil correto dela.
Este projeto nasce de uma necessidade real observada em contexto operacional: redes de abastecimento com reposição periódica onde a alta frequência de requisições complementares ("extras") não é anomalia gerencial, mas sinal estatístico de que as cotas estão calibradas na média de uma distribuição que não é simétrica. O enquadramento escolhido — dominância estocástica de segunda ordem (SSD) — é deliberado: em vez de comparar políticas por seus valores esperados, compara-as pela distribuição inteira de custos, capturando aversão a risco de forma estrutural. Uma política SSD-dominante é preferível a qualquer agente racional avesso a risco, independentemente da forma exata da função utilidade — propriedade que nenhuma comparação de médias ou medianas pode garantir.

O problema
Organizações que usam cotas fixas calibradas pela média histórica simples produzem sistematicamente dois erros simultâneos:

Falta onde a demanda é volátil — unidades com padrão errático recorrem a extras porque a cota não cobre a cauda da distribuição.
Excesso onde a demanda é previsível — capital imobilizado em itens de baixo coeficiente de variação.

A alta participação de requisições complementares (extras) é o sinal observável de que as cotas estão mal dimensionadas. Este projeto quantifica essa ineficiência, estima seu impacto em termos de capital imobilizado e ruptura de serviço, e propõe cotas redesenhadas com base em evidência estocástica — não em médias pontuais.

Impacto estimado (dataset de demonstração)
No dataset sintético de demonstração, mais de 9 em cada 10 pares têm sua cota vigente dominada estocasticamente em 2ª ordem por pelo menos uma das alternativas propostas (cota conservadora ou cota com buffer).
Em termos operacionais, isso significa que a política vigente produz uma distribuição de custos que, para qualquer agente avesso a risco, é inferior à política alternativa — independentemente de qual seja o peso exato atribuído à falta em relação ao excesso.
Para tornar esse resultado concreto: com a razão de custo adotada no modelo de demonstração (falta/excesso = 3,33:1, refletindo que a ruptura de abastecimento custa aproximadamente 3× mais do que o carregamento de uma unidade adicional), a política com buffer reduz o custo esperado total em ~18% em relação à cota vigente para os pares classificados como tipo Z (alta volatilidade). Para os pares tipo X (baixa volatilidade), a cota conservadora reduz o custo de carregamento sem aumentar o risco de ruptura.

Condicional importante. Esses números são sensíveis à razão de custo adotada. A seção de Limitações detalha como calibrar com os parâmetros reais de cada operação.


Abordagem
O enquadramento é de eficiência alocativa sob incerteza estocástica: o objeto não é prever o consumo pontual, mas avaliar qual política de cota tem a melhor distribuição de custo (falta + excesso) sob a demanda observada, usando dominância estocástica de segunda ordem (SSD).
O modelo deliberadamente não se apresenta como preditivo em cenários com série curta (T < 8 meses). A maturidade analítica cresce conforme o histórico acumulado — a arquitetura em camadas reflete essa progressão.

Arquitetura em 4+2 camadas
CamadaFunçãoEntrega1Consolidação e auditoria do dado01_dataset_consolidado.csv2Caracterização descritiva e tipologia XYZ02_metricas_por_par.csv3Política de cota (conservadora e com buffer)03_politicas_cota.csv4Dominância estocástica entre políticas04_dominancia.csv5Previsão de demanda com gate de confiabilidade05_previsoes.csv6Backtesting walk-forward, retreino e drift06_backtest_walkforward.csv

Dataset sintético
O arquivo data/dataset_sintetico.csv foi gerado por generate_synthetic_data.py com seed fixada (SEED = 20260501) e as seguintes propriedades, calibradas para preservar as propriedades estatísticas relevantes ao problema sem correspondência com nenhuma operação real:
PropriedadeValorUnidades15 (Unidade_01 … Unidade_15)Produtos110 em 3 categorias genéricasPares materiais (μ ≥ 0,5/mês)~600Proporção de extras~21 % do consumo totalTipologia XYZX ≈ 35 % · Y ≈ 33 % · Z ≈ 32 %Período completoJan–Abr/2026 (T = 4 meses)Período parcialMai/2026 (só auditoria)
O arcabouço é replicável a qualquer contexto de alocação com função de custo assimétrica — a substituição do dataset sintético por dados reais requer apenas a padronização das colunas de entrada.

Como executar
bashpip install pandas numpy openpyxl scipy scikit-learn lightgbm statsmodels

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

Garantias de replicabilidade

Reprodutibilidade: seed fixada (SEED = 20260501); duas execuções com o mesmo input produzem outputs idênticos.
Parâmetros versionados: todos os parâmetros do modelo concentrados nos objetos imutáveis CONFIG e MLConfig no topo do script.
Trilha de auditoria: cada transformação registrada em log_auditoria.csv com contagem de linhas antes/depois e justificativa.
Especificação formal: ARQUITETURA.md detalha todas as fórmulas, suficientes para reimplementação independente.


Limitações declaradas
LimitaçãoImpactoT = 4 meses (série curta)Pesos MMP fixos; ML em modo sombraRazão de custos falta/excesso = 1,0/0,3 arbitradaSensibilidade da SSD não calibrada com a operação realFunção k em escada (descontinuidade)Gradiente artificial nos cortes de CVPares tratados como independentesNão captura dependência cruzada entre itensHeterogeneidade entre unidades não controladaUnidades grandes/pequenas no mesmo pooledIntervalo ML assume resíduos i.i.d.Subestima incerteza em séries com autocorrelação
A declaração explícita de limitações não é formalidade: é o que distingue um modelo útil de um modelo perigoso. Cada item acima é um vetor de melhoria com roadmap associado.

Roadmap

Acumular base ≥ 8 meses para ativar modelos ML (Ridge, LightGBM, Poisson GLM)
Calibrar razão de custos com a equipe de gestão
Análise de sensibilidade sobre os parâmetros da função k
Suavizar a função de buffer (eliminar descontinuidade)
Modelar dependência cruzada entre itens correlacionados
Incorporar variável de estoque disponível


Stack
Python · pandas · numpy · openpyxl · scipy · scikit-learn · lightgbm · statsmodels · bootstrap não-paramétrico · dominância estocástica de 2ª ordem (SSD)

Licença
Código sob MIT. Metodologia pode ser citada com atribuição.

Autor
Ryan Gabriel de Lima Nascimento
Estudante de Ciências Atuariais na FEA-USP com formação paralela em Relações Internacionais (Universidade Braz Cubas). Atua na interface entre modelagem estocástica, teoria econômica e aplicação institucional — com experiência em gestão de saúde pública e desenvolvimento de sistemas de priorização clínica baseados em evidência.
Este projeto materializa um dos eixos centrais da minha formação: a convicção de que decisões alocativas mal enquadradas teoricamente geram ineficiências estruturais que nenhum ajuste paramétrico corrige. A escolha por dominância estocástica de segunda ordem como critério de comparação de políticas reflete essa perspectiva — priorizar robustez distribucional sobre otimização pontual é, em última análise, uma posição epistemológica sobre o que significa tomar boas decisões sob incerteza real.
GitHub · LinkedIn · Contato: via perfil GitHub
