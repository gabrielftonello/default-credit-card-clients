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

# Diretório onde os arquivos de imagem serão gravados.
pasta_figuras <- "figuras"
if (!dir.exists(pasta_figuras)) {
  dir.create(pasta_figuras, recursive = TRUE)
}

treinar_modelo_ga <- function(
  X_treino,
  y_treino,
  n_subamostra = 3000,
  maxiter_ga = 50,
  seed_ga = 123,
  popSize = 40,
  pcrossover = 0.8,
  pmutation = 0.1,
  elitism = 2
) {
  # Esta função encapsula a etapa de seleção de variáveis por meio de um
  # algoritmo genético binário. A população é composta por cromossomos
  # que representam subconjuntos de preditores (0 = excluído, 1 = incluído),
  # e o valor de adaptação de cada indivíduo é definido pela área sob a
  # curva ROC (AUC) de um modelo de regressão logística ajustado sobre
  # uma subamostra dos dados. A utilização de uma subamostra controlada
  # e de pesos de classe busca equilibrar custo computacional e robustez
  # frente ao desbalanceamento da base.

  y_treino <- factor(y_treino, levels = c(0, 1))

  n <- nrow(X_treino)
  p <- ncol(X_treino)
  nomes_vars <- colnames(X_treino)

  prop_1 <- mean(y_treino == 1)
  prop_0 <- 1 - prop_1

  pesos <- ifelse(
    y_treino == 1,
    0.5 / prop_1,
    0.5 / prop_0
  )

  set.seed(seed_ga)
  idx_sub <- sample(seq_len(n), size = min(n_subamostra, n))

  X_sub <- X_treino[idx_sub, , drop = FALSE]
  y_sub <- y_treino[idx_sub]
  pesos_sub <- pesos[idx_sub]

  cat(
    sprintf(
      "Subamostra de %d linhas utilizada para avaliar o valor de adaptação do algoritmo genético.\n",
      length(idx_sub)
    )
  )

  fitness_ga <- function(bits) {
    if (sum(bits) < 2L) {
      return(0)
    }

    cols <- which(bits == 1)
    X_sel <- X_sub[, cols, drop = FALSE]
    df_sub <- data.frame(default = y_sub, X_sel)

    fit <- try(
      glm(
        default ~ .,
        family = binomial,
        data = df_sub,
        weights = pesos_sub
      ),
      silent = TRUE
    )
    if (inherits(fit, "try-error")) {
      return(0)
    }

    prob <- try(
      predict(fit, type = "response"),
      silent = TRUE
    )
    if (inherits(prob, "try-error")) {
      return(0)
    }

    roc_obj <- try(
      pROC::roc(y_sub, prob, quiet = TRUE),
      silent = TRUE
    )
    if (inherits(roc_obj, "try-error")) {
      return(0)
    }

    as.numeric(roc_obj$auc)
  }

  modelo_ga <- GA::ga(
    type = "binary",
    fitness = fitness_ga,
    nBits = p,
    popSize = popSize,
    maxiter = maxiter_ga,
    run = 5,
    pmutation = pmutation,
    pcrossover = pcrossover,
    elitism = elitism,
    seed = seed_ga,
    monitor = GA::gaMonitor
  )

  bits_best <- modelo_ga@solution[1, ]
  idx_best <- which(bits_best == 1)
  variaveis_sel <- nomes_vars[idx_best]

  cat(
    "Número de variáveis selecionadas pelo algoritmo genético (subamostra): ",
    length(variaveis_sel),
    "\n",
    sep = ""
  )
  cat("Lista de variáveis selecionadas:\n")
  print(variaveis_sel)

  list(
    GA_obj = modelo_ga,
    variaveis_selecionadas = variaveis_sel,
    idx_subamostra = idx_sub
  )
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

# A seguir, o algoritmo genético é aplicado sobre a matriz de treino
# completa, utilizando a AUC como medida de qualidade dos subconjuntos
# de variáveis. O objetivo é reduzir a dimensionalidade de forma guiada
# pelo desempenho preditivo, mantendo um conjunto parcimonioso de
# preditores para a regressão logística.

resultado_ga <- treinar_modelo_ga(
  X_treino = resultado_dados$X_treino,
  y_treino = resultado_dados$y_treino,
  n_subamostra = 3000,
  maxiter_ga = 50,
  seed_ga = 123,
  popSize = 40,
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

# Com as variáveis selecionadas, é ajustado o modelo de regressão
# logística final. Utiliza-se um esquema de ponderação por classe para
# atenuar o impacto do desbalanceamento entre adimplentes e
# inadimplentes. Este modelo passa a ser a base para o cálculo das
# probabilidades de inadimplência e para a definição de políticas de
# decisão baseadas em limiares.

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

# O bloco seguinte executa a rotina de avaliação padronizada, que
# calcula métricas em treino e teste com base em um limiar inicial e
# também obtém as probabilidades previstas. Essas probabilidades são
# reaproveitadas para construir a tabela de desempenho em função de
# diferentes limiares, permitindo a comparação de cenários de decisão.

avaliacao <- avaliar_modelo(
  modelo = modelo_final,
  X_treino = resultado_dados$X_treino,
  X_teste = resultado_dados$X_teste,
  y_treino = resultado_dados$y_treino,
  y_teste = resultado_dados$y_teste,
  variaveis_selecionadas = variaveis_selecionadas
)

if (!is.null(avaliacao$prob_treino) &&
    !is.null(avaliacao$prob_teste)) {
  prob_treino <- avaliacao$prob_treino
  prob_teste <- avaliacao$prob_teste
} else {
  df_teste_final <- data.frame(
    default = factor(resultado_dados$y_teste, levels = c(0, 1)),
    X_teste_sel
  )
  prob_treino <- predict(modelo_final, type = "response")
  prob_teste <- predict(
    modelo_final,
    newdata = df_teste_final,
    type = "response"
  )
}

y_treino <- resultado_dados$y_treino
y_teste <- resultado_dados$y_teste

# Para documentar a sensibilidade do desempenho do modelo à escolha do
# limiar de decisão, é construída uma tabela que relaciona, em dados de
# treino, cada limiar candidato às métricas de acurácia, sensibilidade,
# especificidade, precisão, F1 e índice de Youden.

calcula_metricas_limiar <- function(y_true, prob, limiar) {
  y_pred <- ifelse(prob > limiar, 1, 0)
  y_true <- factor(y_true, levels = c(0, 1))
  y_pred <- factor(y_pred, levels = c(0, 1))

  cm <- table(Real = y_true, Predito = y_pred)

  tn <- ifelse(!is.na(cm["0", "0"]), cm["0", "0"], 0)
  fp <- ifelse(!is.na(cm["0", "1"]), cm["0", "1"], 0)
  fn <- ifelse(!is.na(cm["1", "0"]), cm["1", "0"], 0)
  tp <- ifelse(!is.na(cm["1", "1"]), cm["1", "1"], 0)

  total <- tn + fp + fn + tp
  acuracia <- ifelse(total > 0, (tp + tn) / total, NA)
  sens <- ifelse((tp + fn) > 0, tp / (tp + fn), NA)
  espec <- ifelse((tn + fp) > 0, tn / (tn + fp), NA)
  precisao <- ifelse((tp + fp) > 0, tp / (tp + fp), NA)

  f1 <- ifelse(
    !is.na(precisao) && !is.na(sens) && (precisao + sens) > 0,
    2 * precisao * sens / (precisao + sens),
    NA
  )

  youden <- ifelse(!is.na(sens) && !is.na(espec), sens + espec - 1, NA)

  data.frame(
    Acuracia = acuracia,
    Sensibilidade = sens,
    Especificidade = espec,
    Precisao = precisao,
    F1 = f1,
    Youden = youden
  )
}

limiares <- seq(0.10, 0.90, by = 0.01)

lista_metricas <- lapply(limiares, function(l) {
  metr <- calcula_metricas_limiar(y_treino, prob_treino, l)
  cbind(Limiar = l, metr)
})

tabela_limiares <- do.call(rbind, lista_metricas)

cat("\nTabela de desempenho por limiar (amostra de treino, primeiras linhas):\n")
print(head(tabela_limiares, 6))

# A partir da tabela de limiares, são definidos dois cenários de tomada
# de decisão. O primeiro adota o limiar que maximiza o F1-score,
# enfatizando um equilíbrio global entre precisão e sensibilidade.
# O segundo impõe uma sensibilidade mínima mais alta, refletindo uma
# postura mais conservadora em relação à detecção de inadimplentes, e
# escolhe, entre os limiares admissíveis, aquele com melhor F1.

idx_max_f1 <- which.max(tabela_limiares$F1)
linha_max_f1 <- tabela_limiares[idx_max_f1, , drop = FALSE]

cat("\nLimiar que maximiza o F1 (cenário de equilíbrio geral):\n")
print(linha_max_f1)

limiar_f1 <- linha_max_f1$Limiar

sensibilidade_minima <- 0.75

cand_agressivo <- subset(
  tabela_limiares,
  Sensibilidade >= sensibilidade_minima
)

if (nrow(cand_agressivo) > 0) {
  idx_agressivo <- which.max(cand_agressivo$F1)
  linha_agressiva <- cand_agressivo[idx_agressivo, , drop = FALSE]
} else {
  idx_youden <- which.max(tabela_limiares$Youden)
  linha_agressiva <- tabela_limiares[idx_youden, , drop = FALSE]
}

cat("\nLimiar escolhido para política mais sensível à inadimplência:\n")
print(linha_agressiva)

limiar_agressivo <- linha_agressiva$Limiar
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

cat("\nCenário 2 – política agressiva (sensibilidade mais elevada):\n")
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
print(linha_agressiva)

# Para cada política definida, o desempenho é reavaliado no conjunto de
# teste, permitindo verificar a capacidade de generalização do modelo
# sob diferentes limiares. São reportadas métricas globais e matrizes de
# confusão, o que facilita a análise dos erros relevantes para a gestão
# de risco.

metr_teste_f1 <- calcula_metricas_limiar(y_teste, prob_teste, limiar_f1)

cat("\nMétricas no conjunto de teste (cenário 1 – limiar de equilíbrio):\n")
print(cbind(Limiar = limiar_f1, metr_teste_f1))

pred_teste_f1 <- ifelse(prob_teste > limiar_f1, 1, 0)
cm_teste_f1 <- table(
  Real = factor(y_teste, levels = c(0, 1)),
  Predito = factor(pred_teste_f1, levels = c(0, 1))
)

cat("\nMatriz de confusão no conjunto de teste (cenário 1 – limiar de equilíbrio):\n")
print(cm_teste_f1)

metr_teste_agressivo <- calcula_metricas_limiar(
  y_teste,
  prob_teste,
  limiar_agressivo
)

cat("\nMétricas no conjunto de teste (cenário 2 – política mais agressiva):\n")
print(cbind(Limiar = limiar_agressivo, metr_teste_agressivo))

pred_teste_agressivo <- ifelse(prob_teste > limiar_agressivo, 1, 0)
cm_teste_agressivo <- table(
  Real = factor(y_teste, levels = c(0, 1)),
  Predito = factor(pred_teste_agressivo, levels = c(0, 1))
)

cat("\nMatriz de confusão no conjunto de teste (cenário 2 – política mais agressiva):\n")
print(cm_teste_agressivo)

# Por fim, é realizada uma validação cruzada k-fold utilizando o limiar
# operacional associado ao cenário agressivo. Essa etapa fornece uma
# estimativa mais estável do desempenho médio do modelo sob a política
# escolhida, reduzindo a dependência de uma única partição treino–teste
# e fortalecendo a argumentação sobre a robustez do procedimento.

resultado_cv <- validacao_cruzada_logistica(
  X_treino = resultado_dados$X_treino,
  y_treino = resultado_dados$y_treino,
  variaveis_selecionadas = variaveis_selecionadas,
  n_folds = 5,
  seed = 123,
  limiar_operacional = limiar_agressivo
)

# A rotina de interpretação dos coeficientes da regressão logística
# complementa a análise preditiva com uma discussão substantiva sobre
# a direção e a magnitude dos efeitos das variáveis, expressos por meio
# de odds ratios. Esse resultado é útil para a leitura de negócio, pois
# permite relacionar diretamente os preditores às chances de ocorrência
# de inadimplência.

interpretacao <- interpretar_modelo_logistico(modelo_final)

cat(
  "\nFim da execução do pipeline de modelagem de crédito ",
  "com seleção de variáveis via algoritmo genético, ",
  "regressão logística e validação cruzada.\n"
)
