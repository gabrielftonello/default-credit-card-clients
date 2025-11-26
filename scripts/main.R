# Este script implementa o pipeline principal do estudo de crédito,
# organizando em uma única rotina todas as etapas do processo analítico.
# São encadeadas as seguintes fases: carregamento de pacotes e funções
# auxiliares, preparação e particionamento da base de dados, análise
# exploratória e inspeção via análise de componentes principais, seleção
# de variáveis com algoritmo genético, ajuste de um modelo de regressão
# logística ponderado, escolha e comparação de políticas de decisão por
# meio de diferentes limiares de probabilidade, avaliação em treino e
# teste, validação cruzada e interpretação dos coeficientes do modelo.

cat(
  "Início da execução do pipeline completo ",
  "(seleção via algoritmo genético, regressão logística com ênfase ",
  "em sensibilidade e precisão, e validação cruzada).\n\n"
)

# As instruções source abaixo centralizam o carregamento das funções
# auxiliares em arquivos separados, seguindo uma organização modular
# do código. Essa abordagem facilita a manutenção e a reutilização das
# rotinas de pré-processamento, análise exploratória, avaliação de
# modelos e interpretação.

source("scripts/pacotes_e_funcoes.R")
source("scripts/carregar_e_preparar_dados.R")
source("scripts/eda_e_pca.R")
source("scripts/avaliacao_e_interpretacao.R")
source("scripts/treinamento_ga.R")

# Caso necessário, outros módulos especializados podem ser adicionados
# aqui via source, como, por exemplo, um arquivo dedicado apenas à
# configuração do algoritmo genético para seleção de variáveis.

# Diretório onde os arquivos de imagem serão gravados. A criação é
# feita de forma programática para garantir que os gráficos gerados
# ao longo do pipeline possam ser armazenados e posteriormente
# incorporados em relatórios e documentos.

pasta_figuras <- "figuras"
if (!dir.exists(pasta_figuras)) {
  dir.create(pasta_figuras, recursive = TRUE)
}

# A chamada abaixo executa o módulo de carregamento e pré-processamento
# dos dados, que inclui leitura da base original, tratamento de valores
# faltantes, atenuação de outliers e padronização das variáveis numéricas.
# Como resultado, obtém-se uma lista com versões limpas dos dados e as
# matrizes de treino e teste prontas para a etapa de modelagem.

resultado_dados <- carregar_e_preprocessar_dados(
  caminho_arquivo = "data/default_credit_card_clients.xls",
  proporcao_treino = 0.7,
  seed = 123
)

dados_limpos <- resultado_dados$dados_limpos

# Nesta etapa, são produzidas estatísticas descritivas e visualizações
# básicas para caracterizar a base (como distribuição da variável alvo e
# comportamento de variáveis de interesse) e, em seguida, realiza-se uma
# análise de componentes principais sobre a matriz de preditores
# normalizados. O objetivo é inspecionar a estrutura de variância e
# observar, de forma exploratória, a separação entre classes em um
# espaço de baixa dimensionalidade.

eda_basica(dados_limpos)

pca_result <- rodar_pca(
  X_treino = resultado_dados$X_treino,
  y_treino = resultado_dados$y_treino
)

cat("\nPrimeiras componentes principais e variância explicada:\n")
print(pca_result$variancia_explicada[1:5])
cat("\n")

# Quando disponível, o objeto de PCA é utilizado para construir um
# gráfico de variância explicada, permitindo visualizar a contribuição
# relativa de cada componente principal na variância total dos dados.

if ("pca_obj" %in% names(pca_result)) {
  plot_variancia_pca(
    pca_obj = pca_result$pca_obj,
    k = 10,
    titulo = "PCA – variância explicada",
    nome_arquivo = "pca_variancia_explicada.png"
  )
}

# Em seguida, aplica-se o algoritmo genético sobre a matriz de treino
# completa, utilizando a AUC como medida de qualidade dos subconjuntos
# de variáveis. O objetivo é reduzir a dimensionalidade de forma guiada
# pelo desempenho preditivo, mantendo um conjunto parcimonioso de
# preditores para a regressão logística.

resultado_ga <- treinar_modelo_ga(
  X_treino = resultado_dados$X_treino,
  y_treino = resultado_dados$y_treino,
  n_subamostra = 4000,
  maxiter_ga = 50,
  seed_ga = 123,
  popSize = 80,
  pcrossover = 0.8,
  pmutation = 0.1,
  elitism = 2
)

if (exists("pasta_figuras", inherits = TRUE)) {
  caminho_ga <- file.path(pasta_figuras, "ga_convergencia.png")
  grDevices::png(filename = caminho_ga, width = 800, height = 600)
  plot(
    resultado_ga$GA_obj,
    main = "Evolução do valor de adaptação no algoritmo genético"
  )
  grDevices::dev.off()
}

plot(
  resultado_ga$GA_obj,
  main = "Evolução do valor de adaptação no algoritmo genético"
)

variaveis_selecionadas <- resultado_ga$variaveis_selecionadas

# Com as variáveis selecionadas pelo algoritmo genético, é ajustado o
# modelo de regressão logística final. Utiliza-se um esquema de
# ponderação por classe para atenuar o impacto do desbalanceamento
# entre adimplentes e inadimplentes. Este modelo passa a ser a base
# para o cálculo das probabilidades de inadimplência e para a definição
# de políticas de decisão baseadas em limiares.

X_treino_sel <- resultado_dados$X_treino[
  ,
  variaveis_selecionadas,
  drop = FALSE
]
X_teste_sel <- resultado_dados$X_teste[
  ,
  variaveis_selecionadas,
  drop = FALSE
]

df_treino_final <- data.frame(
  default = factor(resultado_dados$y_treino, levels = c(0, 1)),
  X_treino_sel
)

prop_1 <- mean(df_treino_final$default == 1)
prop_0 <- 1 - prop_1

pesos_full <- ifelse(
  df_treino_final$default == 1,
  0.5 / prop_1,
  0.5 / prop_0
)

modelo_final <- glm(
  default ~ .,
  family = binomial,
  data = df_treino_final,
  weights = pesos_full
)

# A rotina de avaliação padronizada concentra, em uma única função,
# o cálculo das probabilidades previstas em treino e teste, a construção
# da tabela de métricas em função do limiar e a escolha de um limiar
# operacional que respeita uma sensibilidade mínima pré-definida.
# Essa função devolve tanto a tabela de limiares quanto o limiar
# selecionado, além das métricas agregadas e da matriz de confusão no
# conjunto de teste para o ponto de operação escolhido.

sensibilidade_minima <- 0.75

avaliacao <- avaliar_modelo(
  modelo = modelo_final,
  X_treino = resultado_dados$X_treino,
  X_teste = resultado_dados$X_teste,
  y_treino = resultado_dados$y_treino,
  y_teste = resultado_dados$y_teste,
  variaveis_selecionadas = variaveis_selecionadas,
  sens_min = sensibilidade_minima
)

tabela_limiares    <- avaliacao$tabela_limiares
limiar_agressivo   <- avaliacao$limiar_escolhido
metr_treino_aggr   <- avaliacao$treino
metr_teste_aggr    <- avaliacao$teste
cm_teste_agressivo <- avaliacao$cm_teste
prob_treino        <- avaliacao$prob_treino
prob_teste         <- avaliacao$prob_teste

y_treino <- resultado_dados$y_treino
y_teste  <- resultado_dados$y_teste

# A partir da tabela de limiares, define-se um cenário adicional de
# tomada de decisão baseado no limiar que maximiza o F1-score em treino,
# representando uma política de equilíbrio entre sensibilidade e
# precisão. Esse cenário é comparado à política agressiva, que foi
# obtida a partir da restrição de sensibilidade mínima dentro da
# própria rotina de avaliação padronizada.

idx_max_f1 <- which.max(tabela_limiares$F1)
linha_max_f1 <- tabela_limiares[idx_max_f1, , drop = FALSE]
limiar_f1 <- linha_max_f1$Limiar

cat("\nLimiar que maximiza o F1 (cenário de equilíbrio geral):\n")
print(linha_max_f1)

linha_agressiva <- tabela_limiares[
  which(tabela_limiares$Limiar == limiar_agressivo)[1],
  ,
  drop = FALSE
]

cat("\nLimiar escolhido para política mais sensível à inadimplência:\n")
print(linha_agressiva)

plot_metricas_limiar(
  tabela_limiares = tabela_limiares,
  nome_arquivo = "metricas_limiar_treino.png",
  limiares_destaque = c(limiar_f1, limiar_agressivo),
  titulo = "Métricas em função do limiar – dados de treino"
)

cat("\nDescrição resumida das políticas de crédito avaliadas:\n\n")

cat("Cenário 1 – política de equilíbrio (limiar que maximiza F1):\n")
cat(
  sprintf(
    "Se a probabilidade prevista de inadimplência for superior a %.2f, o crédito é cedido.\n",
    limiar_f1
  )
)
cat(
  sprintf(
    "Se a probabilidade prevista de inadimplência for menor ou igual a %.2f, o crédito é mantido.\n",
    limiar_f1
  )
)
cat("Métricas correspondentes na amostra de treino:\n")
print(linha_max_f1)

cat("\nCenário 2 – política agressiva (sensibilidade mínima definida):\n")
cat(
  sprintf(
    "Sensibilidade mínima alvo: %.2f.\n", sensibilidade_minima
  )
)
cat(
  sprintf(
    "Se a probabilidade prevista de inadimplência for superior a %.2f, o crédito é cedido.\n",
    limiar_agressivo
  )
)
cat(
  sprintf(
    "Se a probabilidade prevista de inadimplência for menor ou igual a %.2f, o crédito é mantido.\n",
    limiar_agressivo
  )
)
cat("Métricas correspondentes na amostra de treino:\n")
print(cbind(Limiar = limiar_agressivo, metr_treino_aggr))

# As políticas definidas em treino são agora avaliadas no conjunto de
# teste. Para o cenário de equilíbrio, recalculam-se as métricas e a
# matriz de confusão com base no limiar que maximiza F1. Para o cenário
# agressivo, reutilizam-se diretamente as métricas e a matriz de
# confusão produzidas pela rotina de avaliação padronizada, que já
# consideram o limiar associado à sensibilidade mínima especificada.

metr_teste_f1 <- calcula_metricas(y_teste, prob_teste, limiar_f1)

cat("\nMétricas no conjunto de teste (cenário 1 – limiar de equilíbrio):\n")
print(cbind(Limiar = limiar_f1, metr_teste_f1))

pred_teste_f1 <- ifelse(prob_teste > limiar_f1, 1, 0)
cm_teste_f1 <- table(
  Real = factor(y_teste, levels = c(0, 1)),
  Predito = factor(pred_teste_f1, levels = c(0, 1))
)

cat("\nMatriz de confusão no conjunto de teste (cenário 1 – limiar de equilíbrio):\n")
print(cm_teste_f1)

cat("\nMétricas no conjunto de teste (cenário 2 – política mais agressiva):\n")
print(cbind(Limiar = limiar_agressivo, metr_teste_aggr))

cat("\nMatriz de confusão no conjunto de teste (cenário 2 – política mais agressiva):\n")
print(cm_teste_agressivo)

# Para facilitar a comparação visual entre as políticas, constrói-se
# uma figura única contendo as matrizes de confusão dos dois cenários
# no conjunto de teste. Em cada painel, além das contagens por classe,
# são exibidas as principais métricas de desempenho (acurácia,
# sensibilidade, especificidade, precisão, F1 e índice de Youden),
# o que fornece uma visão simultânea dos erros e da qualidade global
# de cada política de crédito.

plot_matrizes_confusao(
  lista_matrizes = list(
    "Teste – equilíbrio (limiar F1)"           = cm_teste_f1,
    "Teste – agressiva (sensibilidade mínima)" = cm_teste_agressivo
  ),
  lista_metricas = list(
    "Teste – equilíbrio (limiar F1)"           = metr_teste_f1,
    "Teste – agressiva (sensibilidade mínima)" = metr_teste_aggr
  ),
  titulo = "Matrizes de confusão – conjunto de teste",
  nome_arquivo = "matrizes_confusao_teste.png"
)

# A etapa seguinte realiza validação cruzada K-fold utilizando o limiar
# operacional associado à política agressiva. Essa rotina ajusta o
# modelo em múltiplas partições de treino e avalia o desempenho no fold
# de validação correspondente, sempre empregando o mesmo limiar de
# decisão. As métricas fold a fold são então sintetizadas por meio de
# médias e desvios padrão, fornecendo uma estimativa mais estável e
# robusta do desempenho esperado em novos dados.

resultado_cv <- validacao_cruzada_logistica(
  X_treino = resultado_dados$X_treino,
  y_treino = resultado_dados$y_treino,
  variaveis_selecionadas = variaveis_selecionadas,
  n_folds = 5,
  seed = 123,
  limiar_operacional = limiar_agressivo
)

resumo_cv <- resultado_cv$resumo

plot_metricas_cv(
  resumo_cv = resumo_cv,
  titulo = "Validação cruzada (K-fold) – regressão logística",
  nome_arquivo = "validacao_cruzada_metricas.png"
)

# Por fim, a interpretação dos coeficientes da regressão logística
# complementa a análise preditiva com uma visão substantiva da direção
# e magnitude dos efeitos das variáveis. A função dedicada a essa etapa
# apresenta o resumo do modelo, calcula e exibe os odds ratios e
# produz um gráfico com os principais coeficientes ordenados em função
# da magnitude do efeito, o que facilita a comunicação dos resultados
# para a área de negócio.

interpretacao <- interpretar_modelo_logistico(modelo_final)

cat(
  "\nFim da execução do pipeline de modelagem de crédito ",
  "com seleção de variáveis via algoritmo genético, ",
  "regressão logística e validação cruzada.\n"
)
