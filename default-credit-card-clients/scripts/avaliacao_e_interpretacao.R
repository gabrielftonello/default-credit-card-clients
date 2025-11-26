calcula_metricas <- function(y_true, prob, limiar) {
  # Função auxiliar que calcula métricas de classificação para um limiar
  # específico, a partir das probabilidades previstas e do vetor de rótulos
  # verdadeiros. São derivados acurácia, sensibilidade, especificidade,
  # precisão, F1 e índice de Youden, com base na matriz de confusão.

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

avaliar_modelo <- function(
  modelo,
  X_treino,
  X_teste,
  y_treino,
  y_teste,
  variaveis_selecionadas,
  sens_min = 0.70
) {
  # Esta função avalia um modelo de classificação binária (neste caso, uma
  # regressão logística) a partir de conjuntos de treino e teste. Utilizam-se
  # apenas as variáveis previamente selecionadas pelo algoritmo genético,
  # de forma a refletir uma escolha de atributos guiada por desempenho.
  # Além de calcular métricas para diferentes limiares de decisão, a função
  # escolhe um limiar operacional que prioriza sensibilidade mínima desejada
  # e bom equilíbrio global (via F1), retornando também probabilidades
  # previstas para posterior análise.

  X_tr_sel <- X_treino[, variaveis_selecionadas, drop = FALSE]
  X_te_sel <- X_teste[, variaveis_selecionadas, drop = FALSE]

  df_tr <- data.frame(
    X_tr_sel,
    default = factor(y_treino, levels = c(0, 1))
  )
  df_te <- data.frame(
    X_te_sel,
    default = factor(y_teste, levels = c(0, 1))
  )

  # As linhas seguintes obtêm as probabilidades previstas de inadimplência
  # para os conjuntos de treino e teste. Opta-se por trabalhar com
  # probabilidades contínuas para, em seguida, explorar diferentes limiares
  # de decisão em vez de fixar a regra padrão de 0,5, o que permite adaptar
  # a política de crédito ao custo relativo de erros.
  prob_tr <- predict(modelo, newdata = df_tr, type = "response")
  prob_te <- predict(modelo, newdata = df_te, type = "response")

  y_tr_true <- df_tr$default

  # A seguir, explora-se uma grade de limiares entre 0,10 e 0,90. Essa
  # discretização permite inspecionar o comportamento do modelo em diferentes
  # pontos da curva ROC/PR, possibilitando a escolha de um limiar alinhado
  # à política de risco desejada em vez de adotar um valor arbitrário.
  limiares <- seq(0.10, 0.90, by = 0.01)

  tabela_limiares <- do.call(
    rbind,
    lapply(
      limiares,
      function(l) {
        metr <- calcula_metricas(y_tr_true, prob_tr, l)
        cbind(Limiar = l, metr)
      }
    )
  )

  cat("\nAlguns limiares (primeiras linhas - treino, LOGÍSTICA):\n")
  print(head(tabela_limiares, 6))

  # O bloco seguinte implementa a regra de escolha do limiar operacional.
  # Primeiramente, priorizam-se limiares que atendam a uma sensibilidade
  # mínima especificada pelo usuário (sens_min), refletindo o interesse
  # em reduzir falsos negativos (deixar de identificar inadimplentes).
  # Entre esses candidatos, seleciona-se aquele com maior F1, equilibrando
  # sensibilidade e precisão. Caso nenhum limiar atinja a sensibilidade
  # mínima, recorre-se ao índice de Youden, que sintetiza o compromisso
  # entre sensibilidade e especificidade.
  candidatos <- subset(tabela_limiares, Sensibilidade >= sens_min)

  if (nrow(candidatos) > 0) {
    idx_best <- which.max(candidatos$F1)
    melhor <- candidatos[idx_best, ]
    criterio <- sprintf("Sensibilidade >= %.2f e maior F1", sens_min)
  } else {
    idx_best <- which.max(tabela_limiares$Youden)
    melhor <- tabela_limiares[idx_best, ]
    criterio <- "melhor índice de Youden (sens + espec - 1)"
  }

  limiar_escolhido <- as.numeric(melhor$Limiar)

  cat("\nLogística - Limiar escolhido (", criterio, "):\n", sep = "")
  print(melhor)

  # Uma vez definido o limiar, convertem-se as probabilidades previstas em
  # classes discretas para treino e teste e calculam-se as métricas via
  # calcula_metricas, garantindo consistência com a lógica usada na tabela
  # de limiares e na validação cruzada.
  metr_treino <- calcula_metricas(y_tr_true, prob_tr, limiar_escolhido)

  cat("\nLogística - Métricas no TREINO com limiar ajustado:\n")
  print(metr_treino)

  y_te_true <- df_te$default
  metr_teste <- calcula_metricas(y_te_true, prob_te, limiar_escolhido)

  cat("\nLogística - Métricas no TESTE com limiar ajustado:\n")
  print(metr_teste)

  pred_te <- factor(
    ifelse(prob_te > limiar_escolhido, 1, 0),
    levels = c(0, 1)
  )
  cm_teste <- table(Real = y_te_true, Predito = pred_te)
  cat("\nLogística - Matriz de confusão (Teste, limiar ajustado):\n")
  print(cm_teste)

  # Finalmente, são traçadas as curvas ROC para treino e teste, com o
  # objetivo de inspecionar visualmente a capacidade discriminatória do
  # modelo e eventuais sinais de overfitting, ao comparar o desempenho
  # nos dois subconjuntos.
  plot_curva_roc(
    y_verdadeiro = y_treino,
    prob_positiva = prob_tr,
    nome_arquivo = "roc_treino.png",
    titulo = "Curva ROC - Treino"
  )

  plot_curva_roc(
    y_verdadeiro = y_teste,
    prob_positiva = prob_te,
    nome_arquivo = "roc_teste.png",
    titulo = "Curva ROC - Teste"
  )

  # A função devolve um objeto de lista contendo a tabela completa de
  # limiares, o limiar escolhido, as métricas agregadas e as probabilidades
  # previstas. Essa estrutura de dados permite reutilização posterior,
  # tanto para relatórios quanto para análises de sensibilidade.
  list(
    tabela_limiares = tabela_limiares,
    limiar_escolhido = limiar_escolhido,
    treino = metr_treino,
    teste = metr_teste,
    cm_teste = cm_teste,
    prob_treino = prob_tr,
    prob_teste = prob_te
  )
}

interpretar_modelo_logistico <- function(modelo) {
  cat("\nResumo do modelo de regressão logística:\n")
  print(summary(modelo))

  coefs <- coef(modelo)
  odds_ratios <- exp(coefs)

  interpretacao <- data.frame(
    Variavel = names(coefs),
    Coeficiente = coefs,
    OddsRatio = odds_ratios,
    stringsAsFactors = FALSE
  )

  cat("\nOdds Ratios (exp(coef)):\n")
  print(interpretacao)

  cat("\nVariáveis ordenadas por magnitude do efeito (|coef|):\n")
  interpretacao_ord <- interpretacao[
    order(abs(interpretacao$Coeficiente), decreasing = TRUE),
  ]
  print(interpretacao_ord)

  top_n <- min(20L, nrow(interpretacao_ord))
  interpretacao_top <- interpretacao_ord[seq_len(top_n), ]
  interpretacao_top$Variavel <- factor(
    interpretacao_top$Variavel,
    levels = interpretacao_top$Variavel
  )

  graf_coef <- ggplot(interpretacao_top, aes(x = Variavel, y = Coeficiente)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(
      title = "Coeficientes da regressão logística (top 20 por |coef|)",
      x = "Variável",
      y = "Coeficiente (log-odds)"
    ) +
    theme_minimal()

  print(graf_coef)

  if (exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, "logistica_coeficientes_top20.png")
    ggsave(caminho, graf_coef, width = 8, height = 6)
  }

  invisible(interpretacao_ord)
}

validacao_cruzada_logistica <- function(
  X_treino,
  y_treino,
  variaveis_selecionadas,
  n_folds = 5,
  seed = 123,
  limiar_operacional = 0.48
) {
  # Esta função implementa validação cruzada K-fold para o modelo de
  # regressão logística, utilizando somente as variáveis selecionadas
  # pelo algoritmo genético. A estratégia consiste em particionar o
  # conjunto de treino em K subconjuntos mutuamente exclusivos e, em
  # cada iteração, ajustar o modelo em K-1 folds e avaliar o desempenho
  # no fold restante. O objetivo é estimar de forma mais robusta as
  # métricas de generalização associadas a um limiar operacional fixo.

  set.seed(seed)

  y_treino <- factor(y_treino, levels = c(0, 1))

  X_sel <- X_treino[, variaveis_selecionadas, drop = FALSE]

  df_full <- data.frame(X_sel, default = y_treino)

  # A seguir, calculam-se pesos de classe para mitigar o desbalanceamento
  # entre adimplentes e inadimplentes. A ideia é atribuir maior peso à
  # classe minoritária, aproximando a contribuição relativa de cada grupo
  # à função de verossimilhança durante o ajuste da regressão logística.
  prop_1 <- mean(df_full$default == 1)
  prop_0 <- 1 - prop_1

  pesos <- ifelse(
    df_full$default == 1,
    0.5 / prop_1,
    0.5 / prop_0
  )

  n <- nrow(df_full)

  # O vetor de folds é construído de forma aleatória, garantindo que cada
  # observação seja atribuída a exatamente um fold. Essa estrutura de
  # dados é utilizada nos índices lógicos que particionam o conjunto em
  # subconjuntos de treino e validação a cada iteração.
  folds <- sample(rep(1:n_folds, length.out = n))

  # No laço principal percorrem-se os K folds. Em cada iteração, o modelo
  # é ajustado com pesos no subconjunto de treino e, em seguida, são
  # obtidas probabilidades para o fold de validação. As métricas são
  # calculadas no limiar operacional fixado via calcula_metricas, e o AUC
  # é computado por meio do pacote pROC, fornecendo uma medida global de
  # discriminação.
  metricas_folds <- vector("list", n_folds)

  for (k in seq_len(n_folds)) {
    idx_treino <- folds != k
    idx_val <- folds == k

    df_treino_k <- df_full[idx_treino, , drop = FALSE]
    df_val_k <- df_full[idx_val, , drop = FALSE]
    pesos_k <- pesos[idx_treino]

    modelo_k <- glm(
      default ~ .,
      family = binomial,
      data = df_treino_k,
      weights = pesos_k
    )

    prob_val <- predict(modelo_k, newdata = df_val_k, type = "response")
    y_val <- df_val_k$default

    metr_k <- calcula_metricas(y_val, prob_val, limiar_operacional)

    roc_k <- pROC::roc(y_val, prob_val, quiet = TRUE)
    metr_k$AUC <- as.numeric(roc_k$auc)
    metr_k$Fold <- k

    metricas_folds[[k]] <- metr_k
  }

  metricas_folds_df <- do.call(rbind, metricas_folds)

  # O bloco a seguir sintetiza os resultados da validação cruzada por meio
  # de médias e desvios padrão das métricas em todos os folds. Essa visão
  # agregada fornece uma estimativa mais estável do desempenho esperado
  # do modelo em novos dados, reduzindo a dependência de uma única
  # partição treino-teste.
  col_metricas <- c(
    "Acuracia",
    "Sensibilidade",
    "Especificidade",
    "Precisao",
    "F1",
    "Youden",
    "AUC"
  )

  medias <- sapply(
    metricas_folds_df[, col_metricas, drop = FALSE],
    mean,
    na.rm = TRUE
  )
  desvios <- sapply(
    metricas_folds_df[, col_metricas, drop = FALSE],
    sd,
    na.rm = TRUE
  )

  resumo <- data.frame(
    Metrica = col_metricas,
    Media = as.numeric(medias),
    Desvio = as.numeric(desvios)
  )

  cat("\nValidação cruzada K-fold (Regressão Logística):\n")
  cat(sprintf("- Folds: %d\n", n_folds))
  cat(sprintf("- Limiar operacional avaliado: %.2f\n\n", limiar_operacional))
  print(resumo)

  invisible(list(
    metricas_fold_a_fold = metricas_folds_df,
    resumo = resumo,
    n_folds = n_folds,
    limiar_operacional = limiar_operacional
  ))
}
