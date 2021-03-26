# 6BSGI_John_Deere_ESALQ
Repositório com os códigos e dados utilizados para o desafio da John Deere/ESALQ no 6º Workshop de Soluções Matemáticas para Problemas Industriais - CeMEAI - USP

### Requerimentos
- numpy
- pandas
- sklearn

1. O arquivo `feat_eng.py` cria o pipeline para e avalia as métricas de avaliação no conjunto de calibração
2. O arquivo `generate_maps.py` usa os pipelines salvos pelo script anterior para interpolar novas malhas
3. O arquivo `feat_importance.py` estima a importância das variáveis (de acordo com o modelo RF)
4. O arquivo `plot_maps.py` cria scatter plots dos maps interpolados (demorado, dependendo da quantidade de pontos)

### Pastas
- `data`: dados de entrada (calibração e validação)
- `malhas`: malhas geradas para interpolação (com base nos dados georreferenciados)
- `models`: pipelines de processamento salvos para ambos os campos passados pela John Deere/ESALQ
  - Modelos, scalers, feature engineering
  - Esses pipelines são carregados em memória (após sua primeira execução) para evitar processamento desnecessário
- `predictions`: saidas dos modelos (nos conjuntos de validação e nas malhas)
  - os arquivos txt contém métricas de avaliação (desempenho preditivo e tempo de execução -- não de treinamento)

**Apenas os dados de `Ca` foram considerados até o momento.**
