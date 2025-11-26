eda_basica <- function(dados_limpos) {
  # Esta função realiza uma análise exploratória descritiva do conjunto de dados
  # já pré-processado. A ideia é inspecionar dimensões, primeiros registros,
  # estatísticas-resumo e a distribuição da variável alvo, complementando a
  # análise com gráficos simples. Essa etapa fornece uma visão geral da base
  # e ajuda a identificar eventuais assimetrias, desequilíbrios de classe e
  # faixas típicas das variáveis numéricas antes da modelagem preditiva.

  cat("Dimensões do dataset (linhas, colunas):\n")
  print(dim(dados_limpos))

  cat("\nPrimeiras linhas:\n")
  print(head(dados_limpos))

  cat("\nResumo estatístico das variáveis numéricas:\n")
  print(summary(dados_limpos[sapply(dados_limpos, is.numeric)]))

  cat("\nDistribuição da variável alvo (default):\n")
  print(table(dados_limpos$default))
  cat("\nProporções da variável alvo:\n")
  print(prop.table(table(dados_limpos$default)))

  # Além das estatísticas numéricas, são gerados gráficos exploratórios que
  # auxiliam na interpretação da estrutura dos dados. O histograma de idade
  # permite visualizar a concentração de clientes por faixa etária, enquanto
  # o boxplot do limite de crédito por status de inadimplência ilustra a
  # relação entre limite concedido e ocorrência de default. Quando a variável
  # gráfica está disponível, o gráfico é apresentado na tela e, se houver um
  # diretório de figuras definido (pasta_figuras), o arquivo correspondente é
  # gravado para posterior inclusão em relatórios.

  if ("AGE" %in% colnames(dados_limpos)) {
    graf_idade <- ggplot(dados_limpos, aes(x = AGE)) +
      geom_histogram(bins = 30) +
      labs(
        title = "Distribuição da idade dos clientes",
        x = "Idade",
        y = "Frequência"
      )

    print(graf_idade)

    if (exists("pasta_figuras", inherits = TRUE)) {
      caminho_idade <- file.path(pasta_figuras, "eda_hist_idade.png")
      ggsave(
        filename = caminho_idade,
        plot = graf_idade,
        width = 7,
        height = 5
      )
    }
  }

  if ("LIMIT_BAL" %in% colnames(dados_limpos)) {
    graf_limite <- ggplot(dados_limpos, aes(x = default, y = LIMIT_BAL)) +
      geom_boxplot() +
      labs(
        title = "Limite de crédito por status de default",
        x = "Default (0 = não, 1 = sim)",
        y = "Limite de crédito"
      )

    print(graf_limite)

    if (exists("pasta_figuras", inherits = TRUE)) {
      caminho_limite <- file.path(pasta_figuras, "eda_box_limite_por_default.png")
      ggsave(
        filename = caminho_limite,
        plot = graf_limite,
        width = 7,
        height = 5
      )
    }
  }
}

rodar_pca <- function(X_treino, y_treino, n_comp = 2) {
  # Esta função aplica Análise de Componentes Principais (PCA) à matriz de
  # preditores já normalizada utilizada no treino. A PCA é empregada aqui
  # como ferramenta exploratória para investigar a dimensionalidade efetiva
  # do problema, quantificar a variância explicada por cada componente e
  # visualizar a separação entre classes em um espaço de baixa dimensão.
  # Como X_treino já foi centralizada e escalonada no pré-processamento,
  # a chamada de prcomp desabilita a centralização e a padronização internas.

  pca_obj <- prcomp(X_treino, center = FALSE, scale. = FALSE)

  # A variância explicada por componente é extraída para avaliar quanto da
  # informação original é capturada pelas primeiras componentes principais.
  # Isso auxilia na discussão sobre redução de dimensionalidade e na escolha
  # de quantas componentes seriam suficientes para representar o espaço de
  # preditores sem perda relevante de informação.
  var_exp <- summary(pca_obj)$importance[2, ]

  # Em seguida, é construído um data.frame contendo as duas primeiras
  # componentes principais e a classe de default. Essa estrutura de dados
  # serve de base para o gráfico de dispersão, no qual observamos o padrão
  # de distribuição das observações em um plano bidimensional e verificamos
  # visualmente se há alguma separação entre inadimplentes e adimplentes.
  pca_df <- data.frame(
    PC1 = pca_obj$x[, 1],
    PC2 = pca_obj$x[, 2],
    Classe = y_treino
  )

  graf_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Classe)) +
    geom_point(alpha = 0.4) +
    labs(
      title = "Projeção PCA (PC1 x PC2) das amostras",
      x = "Componente principal 1",
      y = "Componente principal 2",
      color = "Default"
    )

  print(graf_pca)

  if (exists("pasta_figuras", inherits = TRUE)) {
    caminho_pca <- file.path(pasta_figuras, "pca_pc1_pc2.png")
    ggsave(
      filename = caminho_pca,
      plot = graf_pca,
      width = 7,
      height = 5
    )
  }

  # A função retorna, de forma organizada, o objeto de PCA completo, o data.frame
  # com as duas primeiras componentes e o vetor de variâncias explicadas. Isso
  # permite reutilizar os resultados em outras etapas do trabalho, como análise
  # adicional de componentes ou comparação com outros métodos de redução de
  # dimensionalidade.
  list(
    pca_obj = pca_obj,
    pca_df = pca_df,
    variancia_explicada = var_exp
  )
}
