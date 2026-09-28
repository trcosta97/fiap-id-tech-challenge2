# Registro de Decisões de Implementação

**Tech Challenge Fase 2 – PosTech IA para Devs**  
**Projeto 1:** Otimização de Regressão Logística via Algoritmo Genético + LLM (Ollama qwen3:8b)

> Cada decisão segue o formato padrão: **Contexto / Opções / Decisão / Justificativa / Consequências**.

---

## Decisão 1 – Foco da Otimização via AG na Regressão Logística

**Data:** 2026-09-28

**Contexto:**  
O Módulo 1 treinou três classificadores (Regressão Logística, Random Forest e XGBoost) sobre o dataset Breast Cancer para prever o desfecho (Alive / Dead). Era necessário escolher qual modelo seria otimizado via Algoritmo Genético no Módulo 2.

**Opções consideradas:**
- Opção A: Otimizar a Regressão Logística (melhor F1 Dead e ROC-AUC do Módulo 1)
- Opção B: Otimizar o XGBoost (segundo melhor F1 Dead)
- Opção C: Otimizar todos os três modelos (comparativo mais amplo)

**Decisão:** Opção A – Regressão Logística.

**Justificativa:**  
A LR obteve os melhores resultados no Módulo 1 (F1=0.3773, ROC-AUC=0.7185, Recall=0.5935), superando RF (F1=0.2857) e XGBoost (F1=0.3456). Além disso, o espaço de hiperparâmetros da LR é bem definido e discretizável (`C`, `penalty`, `solver`, `max_iter`, `tol`), tornando-o ideal para representação cromossômica. Otimizar todos os modelos aumentaria o custo computacional sem ganho metodológico direto.

**Consequências:**
- (+) Espaço de busca compacto e bem documentado (sklearn)
- (+) Comparação direta e justa com o baseline do Módulo 1
- (+) Tempo de execução gerenciável em hardware local
- (-) Não explora a possibilidade de outros modelos superarem a LR após otimização

---

## Decisão 2 – Seleção por Torneio

**Data:** 2026-09-28

**Contexto:**  
Na implementação do AG, é necessário escolher o mecanismo de seleção dos pais para gerar a próxima geração. Os tamanhos de população usados nos experimentos são pequenos (20–40 indivíduos).

**Opções consideradas:**
- Opção A: Seleção por torneio (tournament selection)
- Opção B: Seleção por roleta (roulette wheel / fitness proportionate)
- Opção C: Seleção por ranking (rank-based selection)

**Decisão:** Opção A – Seleção por torneio.

**Justificativa:**  
A seleção por torneio é robusta com populações pequenas porque não requer normalização dos valores de fitness e não sofre pressão seletiva excessiva quando as diferenças entre fitnesses são pequenas (caso típico após poucas gerações de convergência). O parâmetro `tournament_size` permite ajuste direto da pressão seletiva sem alterar o restante do código.

**Consequências:**
- (+) Funciona bem com populações de 20–40 indivíduos
- (+) Parâmetro `tournament_size` é explicitamente configurável por experimento
- (+) Sem risco de dominância precoce de um único indivíduo excelente
- (-) Ligeiramente mais lento que a roleta por envolver comparações entre candidatos

---

## Decisão 3 – Crossover Uniforme

**Data:** 2026-09-28

**Contexto:**  
Era necessário escolher o operador de recombinação (crossover) para gerar filhos a partir de dois pais. O cromossomo é um dicionário com 5 genes independentes entre si (não há dependência posicional).

**Opções consideradas:**
- Opção A: Crossover uniforme (cada gene herdado independentemente com p=0.5)
- Opção B: Crossover de um ponto (corte em posição aleatória do vetor de genes)
- Opção C: Crossover de dois pontos

**Decisão:** Opção A – Crossover uniforme.

**Justificativa:**  
Como os genes são hiperparâmetros independentes (não há relação de ordem ou posição entre `C`, `penalty`, `solver`, `max_iter` e `tol`), o crossover uniforme explora o espaço combinatório de forma mais ampla que os operadores baseados em ponto de corte. Com apenas 5 genes, o crossover de um ponto geraria poucos padrões distintos.

**Consequências:**
- (+) Exploração ampla do espaço combinatório com genes independentes
- (+) Implementação simples e testável gene a gene
- (-) Pode quebrar blocos de construção (building blocks) se existirem interdependências entre genes – aceitável neste caso pois `penalty × solver` é corrigido por `_fix_solver()`

---

## Decisão 4 – Fitness = F1-score da Classe Dead via 5-fold CV

**Data:** 2026-09-28

**Contexto:**  
Definir a função de fitness que guia a evolução do AG. O dataset é desbalanceado (maioria Alive) e o contexto clínico exige sensibilidade à classe minoritária (Dead).

**Opções consideradas:**
- Opção A: F1-score da classe Dead via validação cruzada estratificada (5-fold)
- Opção B: ROC-AUC via validação cruzada
- Opção C: Accuracy
- Opção D: Recall da classe Dead

**Decisão:** Opção A – F1-score (classe Dead), 5-fold Stratified CV.

**Justificativa:**  
O F1 Dead foi a métrica de decisão do Módulo 1, garantindo consistência metodológica. O F1 equilibra precisão e recall da classe minoritária, o que é clinicamente relevante (falsos negativos têm custo alto em oncologia). A validação cruzada estratificada no conjunto de treino fornece uma estimativa mais robusta do que uma avaliação em hold-out único. A Accuracy seria enganosa num dataset desbalanceado.

**Consequências:**
- (+) Métrica idêntica à do Módulo 1, comparação direta e justa
- (+) Respeita o desbalanceamento de classes
- (+) 5-fold CV no treino evita otimização excessiva ao hold-out de teste
- (-) Custo computacional: cada avaliação de fitness treina 5 pipelines completos
- (-) O fitness em CV pode divergir levemente do F1 no teste final

---

## Decisão 5 – LLM Local via Ollama (qwen3:8b) sem Envio de Dados para Nuvem

**Data:** 2026-09-28

**Contexto:**  
O sistema precisa gerar interpretações em linguagem natural dos resultados do modelo. Os dados envolvem registros clínicos de pacientes, o que impõe restrições de privacidade e custo.

**Opções consideradas:**
- Opção A: LLM local via Ollama (qwen3:8b rodando na máquina do usuário)
- Opção B: OpenAI API (GPT-4o ou similar)
- Opção C: Google Gemini API
- Opção D: Hugging Face Inference API

**Decisão:** Opção A – Ollama local com qwen3:8b.

**Justificativa:**  
Dados clínicos de pacientes não devem ser enviados para APIs de terceiros sem processo de anonimização e consentimento formal (LGPD). O Ollama permite execução 100% local, sem custo por token, sem latência de rede e sem riscos de exposição de dados. O modelo qwen3:8b oferece boa capacidade de geração em português com recursos de hardware moderados (GPU/CPU local).

**Consequências:**
- (+) Zero custo de API
- (+) Dados dos pacientes nunca saem do ambiente local (conformidade LGPD)
- (+) Funciona sem conexão à internet após o download do modelo
- (-) Requer instalação prévia do Ollama e download do modelo (~5 GB)
- (-) Qualidade da geração pode ser inferior a modelos maiores em nuvem

---

## Decisão 6 – Prompts em Arquivos .txt Versionados Separadamente do Código

**Data:** 2026-09-28

**Contexto:**  
Os prompts enviados ao LLM definem o comportamento das respostas geradas. Era necessário decidir onde e como armazená-los.

**Opções consideradas:**
- Opção A: Prompts como strings hardcoded no código Python
- Opção B: Prompts em arquivos `.txt` separados, versionados com o repositório
- Opção C: Prompts em banco de dados ou sistema de gerenciamento externo

**Decisão:** Opção B – Arquivos `.txt` em `src/llm/prompts/`.

**Justificativa:**  
Prompts em arquivos separados permitem edição sem alterar o código Python, facilitam revisão via `git diff` e tornam a evolução dos prompts rastreável no histórico do repositório. Hardcoding de strings longas em Python reduz legibilidade e dificulta iteração rápida sobre o prompt.

**Consequências:**
- (+) Histórico de versões do prompt separado do histórico do código
- (+) Edição por pessoas não-programadoras (ex.: redatores clínicos)
- (+) Facilidade de revisão em pull requests
- (-) Requer lógica de carregamento de arquivo em runtime
- (-) O nome do arquivo deve ser estável; renomeações quebram referências no código

---

## Decisão 7 – Modo Mock na LLM para Testes Automatizados

**Data:** 2026-09-28

**Contexto:**  
Os testes automatizados não devem depender de serviços externos. O Ollama pode não estar em execução no ambiente de CI ou na máquina de outro desenvolvedor.

**Opções consideradas:**
- Opção A: Mock via variável de ambiente `LLM_MOCK=true` (retorna resposta fixa sem chamar o Ollama)
- Opção B: Não testar o módulo LLM (excluir da cobertura)
- Opção C: Usar `unittest.mock.patch` em todos os testes que envolvam LLM

**Decisão:** Opção A – modo mock controlado por variável de ambiente.

**Justificativa:**  
O modo mock centraliza a lógica de substituição no próprio `client.py`, evitando dispersão de patches nos arquivos de teste. Qualquer desenvolvedor pode executar a suíte completa de testes com `LLM_MOCK=true` sem instalar nem iniciar o Ollama. Os testes cobrem a lógica de construção do prompt e o fluxo de chamada sem depender de I/O externo.

**Consequências:**
- (+) Testes completamente offline e determinísticos
- (+) Todos os 34 testes passam sem Ollama em execução
- (-) O mock não valida a qualidade real das respostas do LLM
- (-) Requer disciplina para manter o mock sincronizado com a interface real

---

## Decisão 8 – Parada Antecipada: Sem Melhora em 10 Gerações Consecutivas

**Data:** 2026-09-28

**Contexto:**  
O AG precisa de um critério de parada para não executar gerações desnecessárias após a convergência. O número máximo de gerações foi fixado em 30, mas a população pode convergir antes.

**Opções consideradas:**
- Opção A: Parar quando a variação do melhor fitness nas últimas 10 gerações for menor que 1e-5
- Opção B: Parar apenas no limite de gerações (`n_generations`)
- Opção C: Parar quando a diversidade genética da população cair abaixo de um limiar

**Decisão:** Opção A – parada antecipada por estagnação de 10 gerações.

**Justificativa:**  
Nos experimentos realizados, a convergência ocorreu entre as gerações 7 e 12, bem antes das 30 gerações máximas. Continuar após a convergência desperdiça tempo de CPU sem benefício. O critério de variação < 1e-5 é simples, transparente e sem parâmetros adicionais. A janela de 10 gerações é longa o suficiente para evitar paradas prematuras por flutuações numéricas pequenas.

**Consequências:**
- (+) Redução do tempo de execução (experimentos convergiram na geração 12 em média)
- (+) Critério documentado no log para rastreabilidade
- (-) Se o espaço de busca tiver platôs longos, a parada pode ser prematura
- (-) O threshold 1e-5 foi fixado empiricamente; outros domínios podem precisar de ajuste

---

## Decisão 9 – Pip Direto em Vez de Poetry para Gerenciamento de Dependências

**Data:** 2026-09-28

**Contexto:**  
O projeto usa `pyproject.toml` com Poetry como backend de build, mas Poetry não estava disponível no sistema de desenvolvimento. Era necessário um método de instalação funcional imediatamente.

**Opções consideradas:**
- Opção A: Pip direto com `requirements.txt` gerado manualmente a partir do `pyproject.toml`
- Opção B: Instalar Poetry e usar `poetry install`
- Opção C: Conda + `environment.yml`

**Decisão:** Opção A – pip direto com `requirements.txt`.

**Justificativa:**  
Poetry não estava instalado no sistema e sua instalação exigiria configuração adicional de PATH e ambiente. O pip está disponível em qualquer instalação Python padrão. Manter o `pyproject.toml` garante compatibilidade futura com Poetry; o `requirements.txt` é gerado a partir dele para uso imediato.

**Consequências:**
- (+) Instalação imediata sem dependência de ferramentas externas além do pip
- (+) Compatível com qualquer ambiente Python 3.11+
- (+) `pyproject.toml` permanece como fonte de verdade das dependências
- (-) `requirements.txt` precisa ser mantido manualmente sincronizado com o `pyproject.toml`
- (-) Sem lockfile automático gerado pelo Poetry (reprodutibilidade depende de versões pinadas no requirements.txt)

---

## Decisão 10 – Tratamento da Coluna `'T Stage '` com Espaço Trailing no CSV

**Data:** 2026-09-28

**Contexto:**  
O arquivo `data/Breast_Cancer.csv` contém uma coluna com nome `'T Stage '` (com espaço ao final). Sem tratamento, o acesso por nome exato da coluna pode gerar `KeyError` silencioso e o espaço pode causar inconsistências no pipeline de pré-processamento.

**Opções consideradas:**
- Opção A: Renomear a coluna no carregamento via `df.rename()` ou `df.columns = df.columns.str.strip()`
- Opção B: Referenciar sempre o nome com espaço (`'T Stage '`) no código
- Opção C: Corrigir o CSV original e commitar a versão corrigida

**Decisão:** Opção A – strip nas colunas no momento do carregamento em `dataset.py`.

**Justificativa:**  
Corrigir o CSV original (Opção C) quebraria a rastreabilidade com a fonte de dados e poderia afetar outros projetos que usem o mesmo arquivo. Manter o nome com espaço (Opção B) é frágil: qualquer digitação sem espaço gera erro silencioso. O strip no carregamento é uma única linha de código, documentada com comentário, que torna o código robusto sem modificar o dado bruto.

**Consequências:**
- (+) Código robusto: sem KeyErrors por nomes de coluna inconsistentes
- (+) Dataset original preservado sem modificação
- (+) Comportamento explícito e documentado no código
- (-) Qualquer consumidor do CSV que não aplique strip precisará ser alertado sobre este comportamento
