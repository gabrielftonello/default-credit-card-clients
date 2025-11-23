# Esta função implementa um procedimento de seleção de variáveis por meio
# de um algoritmo genético aplicado a um modelo de regressão logística
# binária. O objetivo é encontrar subconjuntos de preditores que resultem
# em bom desempenho na identificação de clientes inadimplentes, com viés
# explícito a favor da classe de default. Para reduzir custo computacional,
# o fitness é avaliado em uma subamostra do conjunto de treino; o modelo
# final, entretanto, é ajustado em todos os dados de treino usando apenas
# as variáveis selecionadas.
#
# Argumentos:
# - X_treino: matriz ou data.frame contendo as variáveis explicativas
#   já pré-processadas para o conjunto de treino.
# - y_treino: vetor de rótulos binários (0/1), representando o status
#   de inadimplência do cliente.
# - peso_classe1: fator de ponderação para a classe 1 (inadimplente),
#   utilizado na regressão logística para incorporar o custo maior de
#   errar clientes em default.
# - maxiter: número máximo de iterações do algoritmo genético.
# - popSize: tamanho da população em cada geração do algoritmo genético.
# - subsample_size: tamanho da subamostra do conjunto de treino usada
#   para avaliar o fitness de cada indivíduo.
# - seed: semente para reprodutibilidade dos resultados.
#
# A função devolve uma lista contendo o objeto do GA ajustado, o modelo
# final de regressão logística treinado com as variáveis selecionadas,
# o vetor de nomes das variáveis escolhidas e a configuração binária
# correspondente ao melhor indivíduo.

treinar_modelo_ga_cv <- function(
  X_treino,
  y_treino,
  peso_classe1   = 5,
  maxiter        = 20,
  popSize        = 25,
  subsample_size = 3000,
  seed           = 123
) {
  set.seed(seed)
  
  n_total <- nrow(X_treino)
  
  # Nesta etapa define-se uma subamostra do conjunto de treino que será
  # utilizada para calcular o fitness dos indivíduos no algoritmo
  # genético. A motivação é acelerar o processo de busca, mantendo uma
  # amostra representativa dos padrões de default sem avaliar o modelo
  # em todas as observações a cada iteração.
  if (subsample_size < n_total) {
    idx_sub <- sample(seq_len(n_total), subsample_size)
  } else {
    idx_sub <- seq_len(n_total)
  }
  
  X_sub <- X_treino[idx_sub, , drop = FALSE]
  y_sub <- y_treino[idx_sub]
  n_sub <- length(y_sub)
  
  cat("Usando subamostra de", n_sub, "linhas para avaliar o fitness do algoritmo genético.\n")
  
  # Aqui são definidos pesos de classe para a regressão logística usada
  # no cálculo do fitness. A classe de inadimplentes (1) recebe um peso
  # maior, refletindo o custo adicional de não detectar um cliente em
  # default. Essa ponderação direciona o GA a priorizar soluções com
  # boa sensibilidade para a classe positiva.
  y_num_sub <- as.numeric(as.character(y_sub))
  pesos_sub <- ifelse(y_num_sub == 1, peso_classe1, 1)
  
  n_bits <- ncol(X_sub)
  nomes_variaveis <- colnames(X_sub)
  
  # A função de fitness avalia cada indivíduo (vetor binário que indica
  # quais variáveis entram no modelo) ajustando uma regressão logística
  # ponderada na subamostra. Em seguida, calcula-se sensibilidade e
  # especificidade e aplica-se uma penalização proporcional ao número de
  # variáveis selecionadas. A combinação linear dessas quantidades gera
  # um escore que favorece modelos capazes de identificar inadimplentes,
  # mas que também mantêm especificidade razoável e complexidade moderada.
  fitness_function <- function(bits) {
    bits <- round(bits)
    idx_sel <- which(bits == 1)
    
    if (length(idx_sel) == 0) {
      return(0)
    }
    
    X_sel_sub <- X_sub[, idx_sel, drop = FALSE]
    df_sub <- data.frame(default = y_sub, X_sel_sub)
    
    mod <- try(
      glm(
        default ~ .,
        data = df_sub,
        family = binomial,
        weights = pesos_sub
      ),
      silent = TRUE
    )
    
    if (inherits(mod, "try-error")) {
      return(0)
    }
    
    prob_sub <- predict(mod, newdata = df_sub, type = "response")
    pred_sub <- ifelse(prob_sub > 0.5, 1, 0)
    pred_sub <- factor(pred_sub, levels = c(0, 1))
    
    metr <- metricas_classificacao(y_sub, pred_sub)
    
    sens <- metr$Sensibilidade
    esp <- metr$Especificidade
    
    k_sel <- length(idx_sel)
    p_total <- n_bits
    penal_comp <- k_sel / p_total
    
    fitness <- 0.7 * sens + 0.3 * esp - 0.2 * penal_comp
    
    if (!is.finite(fitness)) {
      fitness <- 0
    }
    fitness
  }
  
  # Nesta chamada é executado o algoritmo genético propriamente dito,
  # utilizando codificação binária para representar a inclusão ou
  # exclusão de cada variável. O GA explora o espaço de subconjuntos
  # de preditores buscando maximizar o fitness definido acima, com
  # tamanho de população e número de iterações ajustados para um
  # compromisso entre qualidade e tempo de execução.
  ga_obj <- GA::ga(
    type     = "binary",
    fitness  = fitness_function,
    nBits    = n_bits,
    popSize  = popSize,
    maxiter  = maxiter,
    run      = 5,
    seed     = seed,
    parallel = FALSE
  )
  
  melhor_bits <- as.vector(ga_obj@solution[1, ])
  idx_melhor <- which(melhor_bits == 1)
  variaveis_selecionadas <- nomes_variaveis[idx_melhor]
  
  cat(
    "Número de variáveis selecionadas pelo algoritmo genético (subamostra):",
    length(variaveis_selecionadas), "\n"
  )
  cat("Variáveis selecionadas:\n")
  print(variaveis_selecionadas)
  
  # Após definir o subconjunto de variáveis com melhor desempenho na
  # subamostra, o modelo final de regressão logística é ajustado em todo
  # o conjunto de treino, reutilizando a mesma lógica de ponderação por
  # classe. Esta etapa produz o modelo que será de fato utilizado na
  # fase de avaliação e na aplicação da política de crédito.
  X_sel_full <- X_treino[, variaveis_selecionadas, drop = FALSE]
  y_num_full <- as.numeric(as.character(y_treino))
  pesos_full <- ifelse(y_num_full == 1, peso_classe1, 1)
  
  df_treino_final <- data.frame(default = y_treino, X_sel_full)
  
  modelo_final <- glm(
    default ~ .,
    data = df_treino_final,
    family = binomial,
    weights = pesos_full
  )
  
  list(
    ga_obj = ga_obj,
    modelo_final = modelo_final,
    variaveis_selecionadas = variaveis_selecionadas,
    melhor_bits = melhor_bits
  )
}
