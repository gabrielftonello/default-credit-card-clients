# Este script reúne funções para análise de limiares de decisão em modelos
# de classificação binária. A partir das probabilidades previstas, são
# calculadas métricas como acurácia, sensibilidade, especificidade,
# precisão, F1 e índice de Youden em uma grade de limiares. Em seguida,
# são oferecidas rotinas para selecionar limiares de acordo com critérios
# distintos (Youden, F1 global ou sensibilidade mínima) e para produzir
# uma descrição interpretável da política de cessão de crédito associada
# ao limiar escolhido.

grid_metricas_por_limiar <- function(
  y,
  prob,
  seq_limiar = seq(0.1, 0.9, by = 0.01)
) {
  # Esta função constrói uma tabela de métricas de classificação para uma
  # sequência de limiares. Para cada valor de limiar são derivadas as
  # quantidades de verdadeiros positivos, falsos positivos, verdadeiros
  # negativos e falsos negativos, e a partir delas são calculadas acurácia,
  # sensibilidade, especificidade, precisão, F1 e o índice de Youden.
  # A motivação é fornecer uma visão sistemática do compromisso entre erro
  # de aceitação e rejeição do crédito ao variar o ponto de corte da
  # probabilidade prevista de default.
  
  y <- as.numeric(as.character(y))
  
  resultados <- lapply(seq_limiar, function(t) {
    pred <- ifelse(prob > t, 1, 0)
    
    TP <- sum(y == 1 & pred == 1)
    FP <- sum(y == 0 & pred == 1)
    TN <- sum(y == 0 & pred == 0)
    FN <- sum(y == 1 & pred == 0)
    
    total <- TP + FP + TN + FN
    
    acuracia <- ifelse(total > 0, (TP + TN) / total, NA_real_)
    sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
    espec <- ifelse((TN + FP) > 0, TN / (TN + FP), NA_real_)
    precisao <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
    
    if (is.na(sens) || is.na(precisao) || (sens + precisao) == 0) {
      f1 <- NA_real_
    } else {
      f1 <- 2 * precisao * sens / (precisao + sens)
    }
    
    youden <- ifelse(!is.na(sens) && !is.na(espec), sens + espec - 1, NA_real_)
    
    data.frame(
      Limiar = t,
      Acuracia = acuracia,
      Sensibilidade = sens,
      Especificidade = espec,
      Precisao = precisao,
      F1 = f1,
      Youden = youden
    )
  })
  
  do.call(rbind, resultados)
}

escolher_limiar_youden <- function(df_metricas) {
  # Esta função seleciona o limiar de decisão que maximiza o índice de
  # Youden, definido como sensibilidade + especificidade - 1. Esse critério
  # busca um equilíbrio entre a capacidade de identificar inadimplentes e
  # de não penalizar excessivamente bons pagadores, sem impor pesos
  # diferenciados às duas classes.
  
  df <- df_metricas[!is.na(df_metricas$Youden), ]
  if (nrow(df) == 0) {
    warning("Nenhum limiar com índice de Youden definido.")
    return(NULL)
  }
  
  idx <- which.max(df$Youden)
  df[idx, , drop = FALSE]
}

escolher_limiar_sens_prec <- function(
  df_metricas,
  sens_min = 0.75
) {
  # Esta função seleciona um limiar impondo, em primeiro lugar, uma
  # restrição mínima de sensibilidade. Entre os limiares que satisfazem
  # esse requisito, é escolhido aquele que maximiza o F1-score, que
  # sintetiza o compromisso entre sensibilidade e precisão. A motivação é
  # priorizar a identificação de inadimplentes (reduzindo perdas de
  # crédito) sem abrir mão de um mínimo de qualidade nas decisões
  # positivas recomendadas pelo modelo.
  
  df_ok <- df_metricas[
    !is.na(df_metricas$Sensibilidade) &
      !is.na(df_metricas$Precisao) &
      !is.na(df_metricas$F1) &
      df_metricas$Sensibilidade >= sens_min,
  ]
  
  if (nrow(df_ok) == 0) {
    warning(
      paste0(
        "Nenhum limiar atingiu sensibilidade >= ",
        sens_min,
        ". Retornando limiar de melhor F1 global."
      )
    )
    return(escolher_limiar_max_f1(df_metricas))
  }
  
  idx <- which.max(df_ok$F1)
  df_ok[idx, , drop = FALSE]
}

escolher_limiar_max_f1 <- function(df_metricas) {
  # Esta função escolhe o limiar que maximiza o F1-score sem impor
  # restrições prévias à sensibilidade. Essa estratégia é útil quando se
  # deseja um ponto de equilíbrio global entre sensibilidade e precisão,
  # tratando os erros de classificação de forma mais simétrica.
  
  df <- df_metricas[!is.na(df_metricas$F1), ]
  if (nrow(df) == 0) {
    warning("Nenhum limiar com F1 definido.")
    return(NULL)
  }
  
  idx <- which.max(df$F1)
  df[idx, , drop = FALSE]
}

escolher_limiar_sens_min <- function(
  df_metricas,
  sens_min = 0.70
) {
  # Esta função implementa uma estratégia mais simples para seleção de
  # limiar: exige apenas que a sensibilidade atinja um valor mínimo e, a
  # partir dos limiares que atendem a essa condição, escolhe aquele com
  # maior índice de Youden. Ela é mantida como alternativa para comparação
  # conceitual com o critério que utiliza o F1-score de forma explícita.
  
  df_ok <- df_metricas[
    !is.na(df_metricas$Sensibilidade) &
      df_metricas$Sensibilidade >= sens_min,
  ]
  
  if (nrow(df_ok) == 0) {
    return(NULL)
  }
  
  idx <- which.max(df_ok$Youden)
  df_ok[idx, , drop = FALSE]
}

explicar_politica_cesao <- function(
  linha_limiar,
  titulo = NULL
) {
  # Esta função recebe a linha da tabela de métricas correspondente a um
  # determinado limiar e produz uma descrição textual da política de
  # cessão de crédito associada a esse ponto de corte. A ideia é traduzir
  # a regra probabilística em termos de decisão de negócio (conceder ou
  # não o crédito), apresentando simultaneamente as métricas esperadas
  # para facilitar a interpretação gerencial do modelo.
  
  limiar <- linha_limiar$Limiar
  
  acc <- linha_limiar$Acuracia
  sens <- linha_limiar$Sensibilidade
  espec <- linha_limiar$Especificidade
  prec <- if ("Precisao" %in% colnames(linha_limiar)) linha_limiar$Precisao else NA
  f1 <- if ("F1" %in% colnames(linha_limiar)) linha_limiar$F1 else NA
  
  if (!is.null(titulo)) {
    cat("\n", titulo, "\n\n", sep = "")
  }
  
  cat("Política sugerida de cessão de crédito (modelo de regressão logística):\n")
  cat(sprintf("- Se a probabilidade prevista de default for superior a %.2f, recomenda-se ceder o crédito.\n", limiar))
  cat(sprintf("- Se a probabilidade prevista de default for menor ou igual a %.2f, recomenda-se manter o crédito.\n", limiar))
  
  cat("\nMétricas esperadas no conjunto de treino:\n")
  cat(sprintf("  Acurácia:      %.3f\n", acc))
  cat(sprintf("  Sensibilidade: %.3f\n", sens))
  cat(sprintf("  Especificidade:%.3f\n", espec))
  
  if (!is.na(prec)) {
    cat(sprintf("  Precisão:      %.3f\n", prec))
  }
  if (!is.na(f1)) {
    cat(sprintf("  F1-score:      %.3f\n", f1))
  }
}
