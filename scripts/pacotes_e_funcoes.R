# Este script centraliza o carregamento de pacotes e a definição de
# funções auxiliares que são utilizadas ao longo de todo o pipeline.
# A ideia é concentrar em um único ponto os recursos de leitura de dados,
# manipulação, visualização e avaliação de modelos, de forma a tornar o
# restante do código mais enxuto e modular.

pacotes_necessarios <- c(
  "readxl",
  "dplyr",
  "ggplot2",
  "GA",
  "pROC"
)

# O trecho abaixo realiza o carregamento dinâmico dos pacotes declarados
# em 'pacotes_necessarios'. Caso algum pacote ainda não esteja instalado
# no ambiente, ele é instalado com dependências e, em seguida, carregado.
# Essa estratégia visa garantir reprodutibilidade e reduzir problemas de
# execução em ambientes limpos, sem exigir que o usuário instale cada
# pacote manualmente.

for (p in pacotes_necessarios) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

# A função 'plot_curva_roc' produz a curva ROC (Receiver Operating
# Characteristic) a partir dos rótulos verdadeiros e das probabilidades
# previstas para a classe positiva. Em seguida, calcula e exibe a área
# sob a curva (AUC), que resume a capacidade discriminatória global do
# modelo. A implementação utiliza o pacote 'pROC', que oferece recursos
# consolidados para análise de ROC em modelos de classificação. O
# argumento opcional 'cenario' permite explicitar, no título, a política
# ou configuração associada (por exemplo, limiar F1 ou política agressiva).

plot_curva_roc <- function(
  y_verdadeiro,
  prob_positiva,
  nome_arquivo = NULL,
  titulo = "Curva ROC",
  cenario = NULL
) {
  y_num <- as.numeric(as.character(y_verdadeiro))
  roc_obj <- roc(response = y_num, predictor = prob_positiva, quiet = TRUE)

  titulo_plot <- if (!is.null(cenario)) {
    paste0(titulo, " – ", cenario)
  } else {
    titulo
  }

  plot(roc_obj, main = titulo_plot)
  cat("AUC =", auc(roc_obj), "\n")

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    grDevices::png(filename = caminho, width = 800, height = 600)
    plot(roc_obj, main = titulo_plot)
    grDevices::dev.off()
  }
}

# A função 'plot_metricas_limiar' constrói um gráfico de linhas que descreve
# o comportamento de diferentes métricas de desempenho em função do limiar de
# decisão utilizado no classificador. A tabela de entrada deve conter, para
# cada valor de limiar avaliado, colunas com as métricas (por exemplo,
# acurácia, sensibilidade, especificidade, precisão e F1). O objetivo é
# fornecer uma visualização sintética da troca entre sensibilidade,
# especificidade e outras medidas ao variar o ponto de corte, apoiando a
# escolha de uma política operacional de classificação. O argumento
# 'limiares_destaque' permite indicar pontos de operação específicos
# (como o limiar F1 e o limiar agressivo). O argumento 'cenario', quando
# fornecido, é acrescentado ao título para identificar a política analisada.

plot_metricas_limiar <- function(
  tabela_limiares,
  nome_arquivo = NULL,
  limiares_destaque = NULL,
  titulo = "Métricas de desempenho em função do limiar",
  cenario = NULL
) {
  metricas <- c("Acuracia", "Sensibilidade", "Especificidade", "Precisao", "F1")
  metricas <- metricas[metricas %in% colnames(tabela_limiares)]

  df_list <- lapply(metricas, function(m) {
    data.frame(
      Limiar = tabela_limiares$Limiar,
      Metrica = m,
      Valor = tabela_limiares[[m]]
    )
  })

  df_long <- do.call(rbind, df_list)

  titulo_plot <- if (!is.null(cenario)) {
    paste0(titulo, " – ", cenario)
  } else {
    titulo
  }

  graf <- ggplot(df_long, aes(x = Limiar, y = Valor, color = Metrica)) +
    geom_line() +
    labs(
      title = titulo_plot,
      x = "Limiar de decisão",
      y = "Valor da métrica",
      color = "Métrica"
    ) +
    theme_minimal()

  if (!is.null(limiares_destaque)) {
    graf <- graf +
      geom_vline(
        xintercept = limiares_destaque,
        linetype = "dashed"
      )
  }

  print(graf)

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(caminho, graf, width = 8, height = 5)
  }
}

# A função 'plot_matriz_confusao' constrói a visualização de uma matriz de
# confusão 2x2 a partir das contagens de verdadeiros/falsos positivos e
# negativos. Além das frequências por célula, são exibidas as proporções em
# relação ao total de observações, o que facilita a comparação entre cenários
# com tamanhos amostrais distintos. Quando fornecido o data frame de métricas
# associado ao mesmo ponto de operação (acurácia, sensibilidade,
# especificidade, precisão, F1 e índice de Youden), esses valores são
# apresentados em forma resumida no próprio painel, oferecendo uma leitura
# conjunta dos erros de classificação e da qualidade global do modelo. O
# argumento opcional 'cenario' pode ser utilizado para explicitar o tipo de
# política ou limiar empregado (por exemplo, F1 ou agressiva).

plot_matriz_confusao <- function(
  mat_confusao,
  metricas = NULL,
  titulo = "Matriz de confusão",
  nome_arquivo = NULL,
  cenario = NULL
) {
  if (!inherits(mat_confusao, c("matrix", "table"))) {
    stop("mat_confusao deve ser uma matrix ou table 2x2.")
  }

  df <- as.data.frame(as.table(mat_confusao))
  colnames(df) <- c("Real", "Predito", "Freq")

  df$Real    <- as.factor(df$Real)
  df$Predito <- as.factor(df$Predito)
  df$Perc    <- df$Freq / sum(df$Freq)
  df$label   <- sprintf("%d\n(%.1f%%)", df$Freq, 100 * df$Perc)

  titulo_plot <- if (!is.null(cenario)) {
    paste0(titulo, " – ", cenario)
  } else {
    titulo
  }

  label_metricas <- NULL
  if (!is.null(metricas) && is.data.frame(metricas) && nrow(metricas) == 1) {
    m <- metricas[1, ]
    label_metricas <- sprintf(
      "Acc = %.3f  |  Sens = %.3f  |  Esp = %.3f\nPrec = %.3f  |  F1 = %.3f  |  Youd = %.3f",
      m$Acuracia, m$Sensibilidade, m$Especificidade,
      m$Precisao, m$F1, m$Youden
    )
  }

  graf <- ggplot(df, aes(x = Predito, y = Real, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = label), size = 4) +
    scale_fill_gradient(low = "grey90", high = "steelblue") +
    labs(
      title = titulo_plot,
      x = "Classe predita",
      y = "Classe real",
      fill = "Contagem"
    ) +
    coord_equal() +
    theme_minimal()

  if (!is.null(label_metricas)) {
    df_metrics <- data.frame(
      x = 1.5,
      y = Inf,
      label = label_metricas
    )

    graf <- graf +
      geom_text(
        data = df_metrics,
        aes(x = x, y = y, label = label),
        inherit.aes = FALSE,
        vjust = 1.1,
        size = 3
      )
  }

  print(graf)

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(filename = caminho, plot = graf, width = 6, height = 5)
  }
}

# A função 'plot_matrizes_confusao' generaliza a visualização anterior para um
# conjunto de cenários, organizando múltiplas matrizes de confusão em painéis
# (facets). Cada cenário é identificado por um rótulo fornecido na lista de
# entrada. As contagens e proporções por célula são exibidas em cada painel
# e, opcionalmente, são incluídas as métricas numéricas associadas ao cenário
# correspondente (acurácia, sensibilidade, especificidade, precisão, F1 e
# índice de Youden). Essa construção favorece a comparação direta entre
# políticas de decisão alternativas em termos de erros de classificação e
# desempenho global. Recomenda-se que os rótulos dos cenários já indiquem
# se se trata, por exemplo, de limiar F1 ou política agressiva.

plot_matrizes_confusao <- function(
  lista_matrizes,
  lista_metricas = NULL,
  titulo = "Matrizes de confusão",
  nome_arquivo = NULL
) {
  if (!is.list(lista_matrizes) || length(lista_matrizes) == 0) {
    stop("lista_matrizes deve ser uma lista não vazia de matrices/tables 2x2.")
  }

  if (is.null(names(lista_matrizes))) {
    nomes <- paste0("Cenário ", seq_along(lista_matrizes))
  } else {
    nomes <- names(lista_matrizes)
    nomes[nomes == ""] <- paste0("Cenário ", which(nomes == ""))
  }

  dfs <- Map(
    f = function(mat, nome) {
      if (!inherits(mat, c("matrix", "table"))) {
        stop("Todos os elementos de lista_matrizes devem ser matrix ou table.")
      }
      df <- as.data.frame(as.table(mat))
      colnames(df) <- c("Real", "Predito", "Freq")

      df$Real    <- as.factor(df$Real)
      df$Predito <- as.factor(df$Predito)
      df$Perc    <- df$Freq / sum(df$Freq)
      df$label   <- sprintf("%d\n(%.1f%%)", df$Freq, 100 * df$Perc)
      df$Cenario <- nome
      df
    },
    lista_matrizes,
    nomes
  )

  df_all <- do.call(rbind, dfs)

  graf <- ggplot(df_all, aes(x = Predito, y = Real, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = label), size = 3) +
    scale_fill_gradient(low = "grey90", high = "steelblue") +
    labs(
      title = titulo,
      x = "Classe predita",
      y = "Classe real",
      fill = "Contagem"
    ) +
    coord_equal() +
    facet_wrap(~ Cenario) +
    theme_minimal()

  if (!is.null(lista_metricas)) {
    if (!is.list(lista_metricas)) {
      stop("lista_metricas deve ser uma lista (uma entrada por cenário).")
    }

    if (is.null(names(lista_metricas))) {
      names(lista_metricas) <- nomes
    }

    df_metrics_plot <- do.call(
      rbind,
      lapply(nomes, function(nm) {
        metr <- lista_metricas[[nm]]
        if (is.null(metr) || !is.data.frame(metr) || nrow(metr) == 0) {
          return(NULL)
        }
        m <- metr[1, ]
        label_metricas <- sprintf(
          "Acc = %.3f  |  Sens = %.3f  |  Esp = %.3f\nPrec = %.3f  |  F1 = %.3f  |  Youd = %.3f",
          m$Acuracia, m$Sensibilidade, m$Especificidade,
          m$Precisao, m$F1, m$Youden
        )
        data.frame(
          Cenario = nm,
          x = 1.5,
          y = Inf,
          label = label_metricas,
          stringsAsFactors = FALSE
        )
      })
    )

    if (!is.null(df_metrics_plot) && nrow(df_metrics_plot) > 0) {
      graf <- graf +
        geom_text(
          data = df_metrics_plot,
          aes(x = x, y = y, label = label),
          inherit.aes = FALSE,
          vjust = 1.1,
          size = 3
        )
    }
  }

  print(graf)

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(filename = caminho, plot = graf, width = 10, height = 5)
  }
}

# A função 'plot_variancia_pca' representa a variância explicada por cada
# componente principal e a variância acumulada, permitindo avaliar quantos
# componentes são necessários para reter uma proporção desejada da variância
# total dos dados. Esse tipo de gráfico é útil para motivar reduções de
# dimensionalidade e escolhas de k em análises de componentes principais.

plot_variancia_pca <- function(
  pca_obj,
  k = NULL,
  titulo = "Variância explicada pelas componentes principais",
  nome_arquivo = NULL
) {
  if (is.null(pca_obj$sdev)) {
    stop("pca_obj deve ser um objeto retornado por prcomp ou princomp.")
  }

  var_tot <- pca_obj$sdev^2
  prop    <- var_tot / sum(var_tot)

  df <- data.frame(
    Componente          = seq_along(prop),
    Proporcao           = prop,
    ProporcaoAcumulada  = cumsum(prop)
  )

  if (!is.null(k)) {
    df <- df[seq_len(min(k, nrow(df))), ]
  }

  df$Nome <- paste0("PC", df$Componente)

  graf <- ggplot(df, aes(x = Componente)) +
    geom_col(aes(y = Proporcao), alpha = 0.7) +
    geom_line(aes(y = ProporcaoAcumulada), group = 1) +
    geom_point(aes(y = ProporcaoAcumulada)) +
    scale_x_continuous(
      breaks = df$Componente,
      labels = df$Nome
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title = titulo,
      x = "Componente principal",
      y = "Proporção da variância (e acumulada)"
    ) +
    theme_minimal()

  print(graf)

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(filename = caminho, plot = graf, width = 8, height = 5)
  }
}

# A função 'plot_metricas_cv' sintetiza visualmente os resultados da
# validação cruzada a partir de um data frame com métricas, médias e
# desvios padrão. Cada barra representa a média da métrica em todos os
# folds, enquanto as barras de erro indicam a variabilidade (±1 desvio).
# Esse tipo de gráfico permite avaliar, de forma conjunta, a qualidade
# e a estabilidade do modelo sob uma determinada configuração operacional
# (por exemplo, um limiar agressivo). O argumento 'cenario' pode ser
# utilizado para deixar explícita a política avaliada (como F1 ou agressiva).

plot_metricas_cv <- function(
  resumo_cv,
  titulo = "Validação cruzada - resumo de métricas",
  nome_arquivo = NULL,
  cenario = NULL
) {
  col_necessarias <- c("Metrica", "Media", "Desvio")
  if (!all(col_necessarias %in% colnames(resumo_cv))) {
    stop("resumo_cv deve conter as colunas: 'Metrica', 'Media' e 'Desvio'.")
  }

  df <- resumo_cv
  df$Metrica <- factor(df$Metrica, levels = rev(df$Metrica))

  titulo_plot <- if (!is.null(cenario)) {
    paste0(titulo, " – ", cenario)
  } else {
    titulo
  }

  graf <- ggplot(df, aes(x = Metrica, y = Media)) +
    geom_col() +
    geom_errorbar(
      aes(ymin = Media - Desvio, ymax = Media + Desvio),
      width = 0.2
    ) +
    coord_flip() +
    labs(
      title = titulo_plot,
      x = "Métrica",
      y = "Valor médio"
    ) +
    theme_minimal()

  print(graf)

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(filename = caminho, plot = graf, width = 7, height = 5)
  }
}

# A função 'plot_importancia_logistica' organiza os coeficientes da regressão
# logística em termos de magnitude (|coef|), destacando as variáveis com maior
# impacto nas chances de inadimplência. O gráfico diferencia coeficientes
# positivos e negativos, indicando, de forma qualitativa, quais variáveis
# aumentam ou reduzem o risco. O parâmetro top_n controla quantas variáveis
# são exibidas, priorizando comunicação clara com a área de negócio.

plot_importancia_logistica <- function(
  tabela_efeito,
  top_n = 20,
  titulo = "Importância das variáveis (regressão logística)",
  nome_arquivo = NULL
) {
  col_necessarias <- c("Variavel", "Coeficiente")
  if (!all(col_necessarias %in% colnames(tabela_efeito))) {
    stop("tabela_efeito deve conter as colunas 'Variavel' e 'Coeficiente'.")
  }

  df <- tabela_efeito
  df$Importancia <- abs(df$Coeficiente)

  df <- df[order(-df$Importancia), ]
  top_n <- min(top_n, nrow(df))
  df <- df[seq_len(top_n), ]

  df$Sinal <- ifelse(df$Coeficiente >= 0, "Aumenta risco", "Reduz risco")
  df$Variavel <- factor(df$Variavel, levels = rev(df$Variavel))

  graf <- ggplot(df, aes(x = Variavel, y = Coeficiente, fill = Sinal)) +
    geom_col() +
    coord_flip() +
    labs(
      title = titulo,
      x = "Variável",
      y = "Coeficiente logístico",
      fill = "Efeito"
    ) +
    theme_minimal()

  print(graf)

  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(filename = caminho, plot = graf, width = 8, height = 6)
  }
}
