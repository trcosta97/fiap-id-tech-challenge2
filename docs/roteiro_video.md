# Roteiro do Vídeo – Tech Challenge Fase 2
**Duração total estimada: ~12 minutos**  
Navegação: setas ← → do teclado nos slides HTML (`docs/apresentacao.html`).

---

## SLIDE 1 – Capa *(~30 seg)*

> "Olá! Meu nome é Thiago Costa, e esse é o Tech Challenge da Fase 2 do curso PosTech de IA para Devs da FIAP.
>
> O projeto combina duas tecnologias que vimos nessa fase: **Algoritmos Genéticos** para otimizar modelos de machine learning, e **LLMs** para transformar os resultados em laudos que um profissional de saúde consegue entender.
>
> O contexto é um hospital universitário que precisa melhorar seus modelos de diagnóstico oncológico."

---

## SLIDE 2 – Contexto e Problema *(~1 min)*

> "O dataset que usamos é o **Breast Cancer**, com 4024 registros de pacientes com câncer de mama. Cada paciente tem 16 variáveis clínicas, e o objetivo do modelo é prever o desfecho: **Alive ou Dead**.
>
> O principal desafio aqui é o **desbalanceamento**: cerca de 85% dos casos são Alive e só 15% são Dead. Mas justamente a classe Dead é a mais importante clinicamente — se o modelo ignora esses casos, ele pode parecer bom na métrica de acurácia mas falhar onde mais importa.
>
> Por isso, ao longo do projeto, vamos focar no **F1 da classe Dead** e no **ROC-AUC** como métricas principais.
>
> E antes de avançar: qualquer saída da LLM que mostrarmos aqui é de **apoio à decisão** — não substitui o julgamento do profissional de saúde."

---

## SLIDE 3 – Dataset *(~45 seg)*

> "O pré-processamento segue exatamente o mesmo pipeline do Módulo 1: **StandardScaler** nas variáveis numéricas, **OrdinalEncoder** nas categóricas, e removemos a coluna 'Survival Months' porque ela causa data leakage — o modelo não teria essa informação no momento do diagnóstico.
>
> Um detalhe que deu trabalho: a coluna 'T Stage' no CSV original tem um espaço no final do nome. Isso causaria erro silencioso no pipeline, então tratamos com um strip durante o carregamento.
>
> O split é 80/20, estratificado por classe, garantindo que a proporção de Dead seja igual no treino e no teste."

---

## SLIDE 4 – Módulo 1: Baseline *(~1 min 30 seg)*

> "No Módulo 1 treinamos três classificadores com configurações padrão: Regressão Logística, Random Forest e XGBoost. Todos com class_weight='balanced' para compensar o desbalanceamento.
>
> Olhando a tabela: o Random Forest tem a maior acurácia, 82.6%, e isso é uma armadilha. O Recall Dead dele é só 0.22 — ou seja, ele deixa passar quase 78% dos casos de óbito, classificando esses pacientes como Alive. Em triagem oncológica isso é clinicamente inaceitável.
>
> A **Regressão Logística** tem o melhor equilíbrio: maior F1 Dead, maior ROC-AUC e maior Recall Dead. Por isso ela foi escolhida para a otimização via AG.
>
> Aqui já temos os nossos **valores de referência baseline**: F1 Dead de 0.3773 e ROC-AUC de 0.7185."

---

## SLIDE 5 – Seção AG *(~10 seg)*

> "Vamos agora para o Módulo 2: o Algoritmo Genético."

---

## SLIDE 6 – Representação e Operadores *(~1 min 30 seg)*

> "O AG trata cada configuração de hiperparâmetros da Regressão Logística como um **cromossomo com 5 genes**: o parâmetro C de regularização, o tipo de penalidade, o solver, o número máximo de iterações e a tolerância de convergência.
>
> O espaço de busca total é de 720 combinações, mas tem uma restrição importante: nem toda combinação de penalty e solver é válida no scikit-learn. L1, por exemplo, só funciona com o solver 'saga' ou 'liblinear'. Por isso implementamos uma função chamada _fix_solver que é aplicada toda vez que um indivíduo passa por crossover ou mutação — garantindo que a população nunca tenha indivíduos inválidos.
>
> O loop evolutivo funciona assim: avalia o fitness de toda a população, preserva os melhores via elitismo, depois seleciona pais por torneio, aplica crossover uniforme e mutação por substituição. A parada antecipada ocorre se não houver melhora nas últimas 10 gerações.
>
> A **função fitness** é a média do F1 Dead em 5-fold cross-validation no conjunto de treino — nunca vemos o conjunto de teste durante a otimização."

---

## SLIDE 7 – Os 3 Experimentos *(~1 min)*

> "Rodamos 3 experimentos com configurações diferentes do AG.
>
> No **Experimento 1** usamos uma configuração base: população de 20, mutação de 10%. Convergiu na geração 12 e encontrou C=0.1 com penalidade L1 e solver saga.
>
> No **Experimento 2** aumentamos a taxa de mutação para 30% para explorar mais o espaço de busca. Convergiu mais rápido, na geração 10, e encontrou uma combinação diferente — L2 com lbfgs — mas com CV F1 ligeiramente melhor.
>
> No **Experimento 3** dobramos o tamanho da população para 40 e aumentamos o torneio. Resultado: idêntico ao Experimento 1.
>
> O padrão que chama atenção: **todos os experimentos convergiram para C=0.1**, independente da configuração. Isso mostra que o AG identificou consistentemente que regularização forte é a chave para esse dataset."

---

## SLIDE 8 – Comparativo Final *(~1 min)*

> "Vamos ver o que o AG entregou em termos de métricas no conjunto de teste.
>
> O **ROC-AUC** melhorou em todos os experimentos: de 0.7185 no baseline para até 0.7204. Pode parecer pequeno, mas representa uma melhora real na capacidade discriminativa do modelo.
>
> O **CV F1 médio** também melhorou: de 0.4108 para 0.4156 a 0.4160. Esse é o número que o AG otimizou diretamente, e ele reflete melhor generalização.
>
> O F1 Dead no conjunto de teste ficou ligeiramente abaixo do baseline, o que é esperado: o AG otimiza para validação cruzada, não para o conjunto de teste. Essa é uma diferença entre otimização por CV versus ajuste direto no teste.
>
> Em resumo: o AG encontrou hiperparâmetros com melhor capacidade discriminativa geral, evidenciada pelo ROC-AUC."

---

## SLIDE 9 – Seção LLM *(~10 seg)*

> "Agora a parte de NLP: a integração com LLM para geração de laudos interpretativos."

---

## SLIDE 10 – LLM: Prompt e Exemplo *(~1 min 30 seg)*

> "Usamos o **Ollama com o modelo qwen3:8b** rodando completamente local. Isso é fundamental por dois motivos: custo zero de API, e os dados clínicos dos pacientes nunca saem do ambiente — o que está de acordo com a LGPD.
>
> O prompt está versionado em um arquivo de texto separado, o que facilita iterar sem alterar o código. Ele usa a técnica de **role prompting**: definimos o papel como 'assistente de apoio clínico', passamos os dados do paciente, a predição, as probabilidades e as métricas do modelo, e solicitamos uma resposta estruturada em 5 seções.
>
> Na tela você pode ver um trecho de laudo gerado para um paciente de alto risco. O modelo menciona o estadiamento T3, os linfonodos comprometidos, a ausência de receptores hormonais — que são exatamente os fatores mais relevantes nesse caso. E termina com o aviso de que é apenas apoio à decisão.
>
> Para os testes automatizados, implementamos um modo mock que retorna respostas fixas sem precisar do Ollama instalado — os 50 testes passam sem nenhuma dependência externa."

---

## SLIDE 11 – Arquitetura e Testes *(~45 seg)*

> "A estrutura do projeto segue boas práticas de engenharia de software. O código está organizado em módulos com responsabilidades claras: ga para o algoritmo genético, models para os classificadores, llm para a integração com LLM e monitoring para logging estruturado com Loguru.
>
> Todos os resultados dos experimentos são rastreáveis: cada experimento gera um JSON com a configuração completa, o histórico geração a geração e as métricas finais. Os gráficos de convergência também são salvos automaticamente.
>
> Temos mais de 50 testes automatizados cobrindo desde operadores do AG até o carregamento do dataset e os pipelines de treinamento. Tudo reprodutível com seed 42."

---

## SLIDE 12 – Conclusão *(~45 seg)*

> "Para fechar: o Algoritmo Genético cumpriu seu papel. Encontrou hiperparâmetros com melhor ROC-AUC e melhor generalização em cross-validation, e fez isso de forma robusta — configurações bem diferentes chegaram ao mesmo resultado.
>
> A LLM local entrega um nível de interpretabilidade que vai além dos números, gerando laudos que um clínico consegue ler e usar no seu raciocínio. Tudo isso sem expor dados de pacientes.
>
> Os próximos passos naturais seriam uma avaliação formal dos laudos por oncologistas, e expandir o AG para otimizar também outros modelos além da Regressão Logística.
>
> O repositório está disponível no GitHub, link na tela. Obrigado!"

---

## Checklist antes de gravar

- [ ] Abrir `docs/apresentacao.html` no navegador em tela cheia (F11)
- [ ] Testar navegação com as setas do teclado
- [ ] Ter o terminal pronto para mostrar `python experiments/run_experiments.py` ao vivo (opcional, entre slides 7 e 8)
- [ ] Ter o notebook `notebooks/demo_pipeline.ipynb` aberto para demonstrar a LLM ao vivo (opcional, no slide 10)
- [ ] Verificar microfone e qualidade de áudio antes de começar
- [ ] Vídeo deve ter no máximo 15 minutos — esse roteiro está em ~12 min para dar margem
