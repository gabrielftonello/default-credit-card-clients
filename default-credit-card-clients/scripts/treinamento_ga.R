# Função de seleção de variáveis via algoritmo genético para regressão logística.

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
