# Desafio da Disciplina de Aprendizado Profundo

A entrega desse trabalho é opcional, embora altamente recomendada, e pode garantir ao aluno até 1 ponto extra na nota da prova.

## Intro 

O desafio consiste em implementar e treinar uma Rede Neural capaz de realizar operações matemáticas usando imagens de dígitos como entrada. Dessa forma, a rede deverá receber a imagem de um número A (entre 0 e 9), a imagem de um número B (entre 0 e 9), uma operação matemática (soma = '+', subtração = '-', multiplicação = '*', divisão = '/'), e prover a resposta da operação como saída. Por exemplo:


| Image A| Image B | Operator | Resultado |
| :--: | :--: | :--: | :--: |
| ![alt text](deep_equation/resources/digit_a.png "Title")     | ![alt text](deep_equation/resources/digit_b.png "Title")     |  +    |  12 |
| ![alt text](deep_equation/resources/digit_a.png "Title")     | ![alt text](deep_equation/resources/digit_b.png "Title")     |  *    |  35 |
| ![alt text](deep_equation/resources/digit_b.png "Title")     | ![alt text](deep_equation/resources/digit_a.png "Title")     |  -    | -2  |
| ![alt text](deep_equation/resources/digit_b.png "Title")     | ![alt text](deep_equation/resources/digit_a.png "Title")     |  /    | 0.71  |

Para facilitar, os dígitos podem ser arrendondados com duas casas decimais. Você pode modelar o problema como classificação, regressão, ou qualquer outro tipo de tarefa que ache interessante. 

Você pode treinar em qualquer framework, o importante é conseguir abrir o modelo treinado como um predictor do pacote `deep_equation`. A interface de implementação é bem simples: você deve escrever um método para carregar o modelo treinado, e outro para fazer as predições. Tem um pacote de testes que vai lhe dizer se a implementação está sintaticamente correta. 

> Importante pensar que nem todas as imagens terão o fundo branco e número preto. As imagens podem vir com diferentes cores, tamanhos e distorções. Então, pense em como aliviar esse tipo de problema. 

## Interface de Entrega 

Você deverá implementar um pacote pip-instalável do Python. Todos os alunos terão acesso a um pacote de exemplo (disponível e *forkable* no github), o qual contém um modelo aleatório que está com a sintaxe de acordo com a entrega. Dessa forma, você pode fazer um *fork* e editar o pacote para conter sua rede neural treinada. OBS: Todos os códigos submetidos serão instalados e avaliados em um test set separado, veja mais detalhes na seção de [Avaliação](#avaliação).

A interface do sistema deverá seguir o seguinte padrão (mais detalhes no readme do pacote `deep_equation`):

```python
from PIL import Image

class BaseNet:

    def predict(self, images_a: List, images_b: List, operators: List, device: str = 'cpu') -> List[float]:
    """
    Make a batch prediction considering a mathematical operator 
    using digits from image_a and image_b.
    Instances from iamges_a, images_b, and operators are aligned:
        - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
    Args: 
        * images_a (List[PIL.Image]): List of RGB PIL Image of any size
        * images_b (List[PIL.Image]): List of RGB PIL Image of any size
        * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
            - invalid options must return `None`
        * device: 'cpu' or 'cuda'
    Return: 
        * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
            [{digit from image_a} {operator} {digit from image_b}]
    """
        # do your magic
        return predicted_number
```

A implementação do seu modelo (*predição do modelo treinado*) deverá ser implementada na classe `StudentModel`.


## Restrições

* Use o pacote de exemplo como base. 
* Você **NÃO** pode usar um sistema de OCR para extrair o dígito. 
* O sistema deve ser capaz de processar imagens RGB de qualquer tamanho (todo o pré-processamento é por sua conta!)
* Você é responsável por coletar dados, otimizar os hiper-parâmetros, arquitetura, treinar o modelo, e disponibilizar o código bem como o modelo. 
* A arquitetura tem que ser end-to-end: uma vez que a imagem entra na Rede Neural, a saída tem que ser o número resultante: isto é, a rede vai ter que aprender a extrair o valor semântico do número, mas também a realizar a operação desejada. 
* É permitida uma etapa de pós-processamento para converter, processar, escalar valores da saída.
* O modelo e o código implementados precisam passar por todos os testes unitários implementados no pacote de exemplo.

## Entrega

A entrega deverá ser um `url` do repositório no github. Você pode salvar os pesos do modelo no próprio repositório se quiser, ou em um sistema de storage como google drive. Entretanto, lembre-se que o modelo deve estar disponível e ser carregado **automaticamente** pela classe de predição. 

## Avaliação

O processo de avaliação é automático, e será efetuado da seguinte forma (com um script parecido com esse):


> Inclusive o processo para você instalar o pacote base da entrega `deep_equation` é exatamente o mesmo. 

```sh
git clone <your_repository>
cd <your_repository>
pip install -e deep_equation
```

```python
from <your_package> import Predictor

# Here the network should be loaded and be ready to predict
# please, use gpu if that is available
predictor = Predictor()
inst_preds = []
for image_a, image_b, operator in zip(ims_a, ims_b, operators):
     inst_pred = predictor.predict(image_a, image_b, operator)
     inst_preds.append(inst_pred)

metrics = calculate_metrics(inst_preds, ground_truth)
store_metrics(metrics)
```

A pontuação da tarefa (entre 0 e 1) será o resultado de uma coleção de métricas (basicamente uma adaptação do erro absoluto médio, que chamarei de `acerto`) calculada em um conjunto de teste que não será liberado aos alunos. 
* Caso o sistema atinga 100% de `acerto`, o aluno recebe 1 ponto extra na prova. 
* Caso o sistema atinga 50% de `acerto` no teste, o aluno vai receber 0.5 pontos extra na prova. 
* Exemplo: aluno tirou 7.5 na prova, fez o trabalho e seu sistema conseguiu 80% de `acerto`. A nota final da prova será: 8.3. *Sounds good, right?*

O aluno vai ter o desafio **zerado** em alguns casos, como: 

* O código não passe nos testes unitários. 
* O sistema gere um erro imprevisto e tenha sua execução interrompida, como por exemplo: estouro de memória (16GB RAM deve ser suficiente).
* Seja detectado plágio na implementação.
* O sistema atingiu 0% de acurácia (o que é praticamente impossível, a menos que exista um erro na implementação).
* O sistema demore mais que 30 minutos para fazer as predições do test set [timeout!]. Tente obter uma média de no máx 2s por instância.

