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
# consolidados para análise de ROC em modelos de classificação.

plot_curva_roc <- function(
  y_verdadeiro,
  prob_positiva,
  nome_arquivo = NULL,
  titulo = "Curva ROC"
) {
  y_num <- as.numeric(as.character(y_verdadeiro))
  roc_obj <- roc(response = y_num, predictor = prob_positiva, quiet = TRUE)
  
  plot(roc_obj, main = titulo)
  cat("AUC =", auc(roc_obj), "\n")
  
  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    grDevices::png(filename = caminho, width = 800, height = 600)
    plot(roc_obj, main = titulo)
    grDevices::dev.off()
  }
}

plot_metricas_limiar <- function(
  tabela_limiares,
  nome_arquivo = NULL,
  limiares_destaque = NULL,
  titulo = "Métricas de desempenho em função do limiar"
) {
  # Esta função constrói um gráfico de linhas que descreve o comportamento
  # de diferentes métricas de desempenho em função do limiar de decisão
  # utilizado no classificador. A tabela de entrada deve conter, para cada
  # valor de limiar avaliado, colunas com as métricas (por exemplo, acurácia,
  # sensibilidade, especificidade, precisão e F1). O objetivo é fornecer uma
  # visualização sintética da troca entre sensibilidade, especificidade e
  # outras medidas ao variar o ponto de corte, apoiando a escolha de uma
  # política operacional de classificação.
  #
  # Opcionalmente, a função permite destacar determinados limiares considerados
  # relevantes (por exemplo, aquele que maximiza o F1 ou que satisfaz uma
  # sensibilidade mínima), desenhando linhas verticais tracejadas nessas
  # posições. Além disso, caso seja fornecido um nome de arquivo e exista
  # no ambiente o diretório de figuras (pasta_figuras), o gráfico gerado é
  # gravado em disco para uso em relatórios e documentos.

  metricas <- c("Acuracia", "Sensibilidade", "Especificidade", "Precisao", "F1")
  metricas <- metricas[metricas %in% colnames(tabela_limiares)]

  # A estrutura de dados é convertida do formato "wide" (uma coluna por
  # métrica) para o formato "long", no qual cada linha representa uma
  # combinação de limiar e métrica. Essa reorganização facilita a utilização
  # da gramática de gráficos do ggplot2, permitindo a criação de uma única
  # figura com múltiplas curvas, cada qual correspondente a uma métrica.
  df_list <- lapply(metricas, function(m) {
    data.frame(
      Limiar = tabela_limiares$Limiar,
      Metrica = m,
      Valor = tabela_limiares[[m]]
    )
  })

  df_long <- do.call(rbind, df_list)

  graf <- ggplot(df_long, aes(x = Limiar, y = Valor, color = Metrica)) +
    geom_line() +
    labs(
      title = titulo,
      x = "Limiar de decisão",
      y = "Valor da métrica",
      color = "Métrica"
    ) +
    theme_minimal()

  # Quando o usuário fornece limiares de interesse, estes são indicados no
  # gráfico por linhas verticais tracejadas. Esse recurso visual destaca
  # pontos de operação específicos e favorece a discussão sobre as métricas
  # obtidas nesses patamares.
  if (!is.null(limiares_destaque)) {
    graf <- graf +
      geom_vline(
        xintercept = limiares_destaque,
        linetype = "dashed"
      )
  }

  print(graf)

  # Se o nome de arquivo foi especificado e o diretório de figuras está
  # definido no ambiente, o gráfico é exportado como imagem. Esse procedimento
  # automatiza a geração de material gráfico para inclusão em relatórios
  # acadêmicos ou documentos técnicos, mantendo a consistência entre o que é
  # exibido no console e o que é armazenado para documentação.
  if (!is.null(nome_arquivo) && exists("pasta_figuras", inherits = TRUE)) {
    caminho <- file.path(pasta_figuras, nome_arquivo)
    ggsave(caminho, graf, width = 8, height = 5)
  }
}