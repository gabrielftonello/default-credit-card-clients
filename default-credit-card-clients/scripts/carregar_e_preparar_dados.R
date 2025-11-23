carregar_e_preprocessar_dados <- function(
  caminho_arquivo = "data/default_credit_card_clients.xls",
  proporcao_treino = 0.7,
  seed = 123
) {
  # Esta função implementa a etapa de preparação dos dados para modelagem
  # preditiva de inadimplência. Nela são realizadas, de forma integrada,
  # as tarefas de leitura do arquivo original, tratamento de tipos,
  # imputação de valores faltantes, atenuação de outliers, divisão em
  # conjuntos de treino e teste e construção de matrizes de preditores
  # padronizadas, que serão utilizadas diretamente pelos modelos.

  # A leitura do arquivo utiliza skip = 1 porque a base original possui uma
  # primeira linha com rótulos genéricos (X1, X2, ...) e uma segunda linha
  # com nomes efetivamente descritivos das variáveis. Ao ignorar a primeira
  # linha, garantimos que os nomes das colunas reflitam o significado real
  # de cada atributo.
  dados_brutos <- readxl::read_excel(caminho_arquivo, skip = 1)

  # A coluna de identificação (ID), quando presente, é removida por não
  # acrescentar informação preditiva relevante. Trata-se de um identificador
  # puramente nominal, que introduziria ruído se mantido como variável de
  # entrada e não contribui para a compreensão do comportamento de crédito.
  if ("ID" %in% colnames(dados_brutos)) {
    dados_brutos <- dados_brutos |>
      dplyr::select(-ID)
  }

  # A variável resposta é renomeada para "default" com o objetivo de
  # simplificar a referência no restante do código e tornar a semântica
  # do modelo mais clara. Essa padronização facilita a construção de
  # fórmulas e a interpretação posterior dos resultados.
  if ("default payment next month" %in% colnames(dados_brutos)) {
    colnames(dados_brutos)[
      colnames(dados_brutos) == "default payment next month"
    ] <- "default"
  }

  # As variáveis categóricas (sexo, escolaridade, estado civil, histórico
  # de pagamento e a própria variável alvo) são convertidas para o tipo
  # fator. Essa escolha é fundamental para que as funções de modelagem
  # reconheçam corretamente a natureza discreta desses atributos e
  # produzam, via codificação dummy, representações adequadas para os
  # algoritmos lineares utilizados.
  cols_fator_basicas <- c("SEX", "EDUCATION", "MARRIAGE", "default")
  cols_pagamento <- paste0("PAY_", c(0, 2:6))

  cols_fatores <- intersect(
    c(cols_fator_basicas, cols_pagamento),
    colnames(dados_brutos)
  )

  dados_brutos[cols_fatores] <- lapply(
    dados_brutos[cols_fatores],
    function(x) factor(x)
  )

  # A variável alvo é explicitamente forçada a ser um fator binário com
  # níveis "0" e "1". Essa padronização evita ambiguidades na ordenação
  # dos níveis e garante que os modelos interpretem corretamente qual
  # categoria corresponde à inadimplência.
  dados_brutos$default <- factor(dados_brutos$default, levels = c(0, 1))

  # O tratamento de valores faltantes é feito de maneira simples e
  # sistemática: para variáveis numéricas utiliza-se a mediana, que é
  # robusta a outliers; para fatores ou caracteres, emprega-se a moda,
  # mantendo a categoria mais frequente. Essa estratégia busca preservar
  # a distribuição original das variáveis sem introduzir valores extremos
  # artificiais.
  tratar_na <- function(coluna) {
    if (any(is.na(coluna))) {
      if (is.numeric(coluna)) {
        valor <- stats::median(coluna, na.rm = TRUE)
        coluna[is.na(coluna)] <- valor
      } else if (is.factor(coluna) || is.character(coluna)) {
        tab <- table(coluna)
        moda <- names(which.max(tab))
        coluna[is.na(coluna)] <- moda
      }
    }
    coluna
  }

  dados_limpos <- as.data.frame(lapply(dados_brutos, tratar_na))

  # A seguir é aplicada uma winsorização simples nas variáveis numéricas,
  # limitando os valores ao intervalo entre os quantis 1% e 99%. A motivação
  # é reduzir o impacto de outliers extremos, que podem distorcer a
  # estimação de parâmetros em modelos lineares, sem recorrer à remoção
  # de observações, o que preserva o tamanho amostral.
  cols_numericas <- names(dados_limpos)[sapply(dados_limpos, is.numeric)]

  for (col in cols_numericas) {
    x <- dados_limpos[[col]]
    q1 <- stats::quantile(x, 0.01, na.rm = TRUE)
    q99 <- stats::quantile(x, 0.99, na.rm = TRUE)
    x[x < q1] <- q1
    x[x > q99] <- q99
    dados_limpos[[col]] <- x
  }

  # A divisão entre conjunto de treino e conjunto de teste é realizada de
  # forma aleatória, respeitando a proporção especificada. O uso de uma
  # semente fixa garante reprodutibilidade, permitindo que diferentes
  # execuções do código produzam a mesma partição e, consequentemente,
  # métricas comparáveis ao longo do trabalho.
  set.seed(seed)
  n <- nrow(dados_limpos)
  idx_treino <- sample(seq_len(n), size = floor(proporcao_treino * n))

  dados_treino <- dados_limpos[idx_treino, ]
  dados_teste <- dados_limpos[-idx_treino, ]

  # Para alimentar os modelos, constrói-se uma matriz de preditores a partir
  # de model.matrix, que expande fatores em variáveis indicadoras (dummies)
  # e remove o intercepto implícito. Em seguida, aplica-se padronização
  # (centralização e escalonamento) com base apenas nas estatísticas do
  # conjunto de treino, o que evita contaminação de informação entre treino
  # e teste e melhora a estabilidade numérica dos algoritmos de otimização.
  X_treino_mm <- stats::model.matrix(default ~ ., data = dados_treino)[, -1]
  X_teste_mm <- stats::model.matrix(default ~ ., data = dados_teste)[, -1]

  medias <- apply(X_treino_mm, 2, mean)
  desvios <- apply(X_treino_mm, 2, sd)

  # Quando uma variável apresenta desvio padrão nulo, substitui-se o
  # denominador por 1 para evitar divisão por zero. Na prática, isso
  # significa que variáveis constantes permanecem inalteradas após a
  # padronização, o que é coerente com a ausência de variabilidade.
  desvios[desvios == 0] <- 1

  X_treino <- scale(X_treino_mm, center = medias, scale = desvios)
  X_teste <- scale(X_teste_mm, center = medias, scale = desvios)

  y_treino <- dados_treino$default
  y_teste <- dados_teste$default

  # A função retorna uma lista estruturada com os principais objetos
  # gerados ao longo do pré-processamento. Dessa forma, é possível
  # reutilizar tanto os dados limpos quanto as matrizes padronizadas
  # e os parâmetros de escala em etapas posteriores, como ajuste de
  # modelos adicionais ou aplicação do pipeline em novos conjuntos de
  # observações.
  list(
    dados_brutos = dados_brutos,
    dados_limpos = dados_limpos,
    dados_treino = dados_treino,
    dados_teste = dados_teste,
    X_treino = as.matrix(X_treino),
    X_teste = as.matrix(X_teste),
    y_treino = y_treino,
    y_teste = y_teste,
    medias = medias,
    desvios = desvios
  )
}
