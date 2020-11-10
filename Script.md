# Script

This program is meant to simulate a very basic neural network. 

What is a neural network?

A neural network works much like the neurons of the human brain and nervous system; sensory input whether by taste, touch, sound, sight, or smell stimulates a certain body response. In a similar way, a computational neural network receives certain input and produces a desired. However, there is one fundamental difference. Our sensory input often renders an involuntary body response, whereas the computer has no idea how to respond to certain input! That's why we have machine learning, which we use to train the computer so we arrive at a desired output. At the heart of a neural network lies three types of layers: the input, hidden, and output layers. These are organized in that order from left to right. Each of these layers contains several \"neurons,\" and for this program, each layer has two neurons. Each neuron of the input layer is linked to each neuron of the hidden layer, and each neuron of the hidden layer is linked to each neuoron of the output layer. 

What exactly do we mean by link? 

A link is represented by a numerical value we call a weight. In this program, the weights are initially set to somewhat random values. 

How do the layers and weights interact with each other?

The layers and weights interact in two primary parts: forward propagation and back propagation. In forward propagation we manipulate the input values with the weights (between the input and hidden layers) to produce hidden values. Then we manipulate the hidden values with the weights (between the hidden and output layers) to produce output values. When we reach our outputs, we compare our results with the desired outputs, which we can call target outputs. For this program, our inputs are 0.05 and 0.10, and the target outputs are 0.01 and 0.99, heuristically close to 0 and 1, or binary true and false. In comparing them, we come up with a number we call the Total Error, which represents how far off our actual outputs are from the targets. Then we move on to backward propagation, where we manipulate now manipulate the weight values to ensure the results are closer to the target. This program implements a number of standard formulas like the logistical formula, the total error formula, the delta rule, and the partial derivative, as well as standard constant values such as the bias and the learning rate. This is complex Calculus even I only learned over the past two years of high school. However, understanding how to apply these formulas and the mathematical theorems that allow us to arrive at each vastly deepens our understanding of neural networks. Many of the important formulas are described in their own unique function, for this program is meant to illustrate the fomulaic applications and show that neural networks aren't as abstract or daunting as we think. Though this program is very very simple relative to the complex applications of neural networks used in the professional world, this acts as a theoretical exercise so that high schoolers like me may understand the basic concepts.");

Review the console output and notice that in 10,000 trials we nearly reach our target output.

## Variables

```python
//establishes input values and the input bias
var i1 = 0.05;
var i2 = 0.10;
var iBias = 0.35;

//establishes weight values between the input and hidden layers
var w1 = 0.15;
var w2 = 0.20;
var w3 = 0.25;
var w4 = 0.30;
var hBias = 0.60;

//establishes weight values between the hidden and output layers
var w5 = 0.40;
var w6 = 0.45;
var w7 = 0.50;
var w8 = 0.55;

//creates arrays that will store the values of the manipulated input numbers
var input = [i1,i2];
var hidden = [];
var output = [];

//creates arrays for convenient multiplication using the weights
var inputToHidden = [w1,w2,w3,w4];
var hiddenToOutput = [w5,w6,w7,w8];

var neuronCount = 0;

var targetOutput1 = 0.01;
var targetOutput2 = 0.99;

var targetOutputList = [targetOutput1,targetOutput2];

//delta rule = deltaRulePart1 * deltaRulePart2 * connecting hidden neuron output
var deltaRulePart1 = []; //total error change with respect to output
var deltaRulePart2 = []; //derivative of logistic function
var deltaNode = []; //part1 * part 2
var learningRateValue = 0.5;
var weightAdjustmentListHiddenToOutput = [];

//partial derivative
var partialDerivativePart1 = [];
var partialDerivativePart2 = [];
var partialDerivativePartProduct = [];
var weightAdjustmentListInputToHidden = [];

```

## Call Functions

```python
introducingNeuralNetwork();
demonstratingNeuralNetworkRound1();
demonstratingNeuralNetworkRound2();
runningNeuralNetwork();
```

## Function Definitions

### Demonstrating Neural Network (Round 1)

```python
function demonstratingNeuralNetworkRound1() {

  layerToLayerForward(input,inputToHidden,iBias,hidden);
  printList(hidden,"Round 1 \t\tHidden Layer Neurons");

  layerToLayerForward(hidden,hiddenToOutput,hBias,output);
  printList(output,"Round 1 \t\tOutput Layer Neurons");

  console.log(" ");
  console.log("Round 1 \t\tOutput Total Error:");
  console.log("\t\t\t" + totalErrorFunction(output,targetOutputList));

  deltaRulePart1Function(output,targetOutputList);
  deltaRulePart2Function(output);
  deltaNodeFunction(deltaRulePart1,deltaRulePart2);
  printList(deltaNode,"Round 1 \t\tDelta Node Values for o1 & o2");

  var newW5 = hiddenToOutput[0] - (learningRateValue * (deltaNode[0] * hidden[0]));
  weightAdjustmentListHiddenToOutput.push(newW5);
  var newW6 = hiddenToOutput[1] - (learningRateValue * (deltaNode[0] * hidden[1]));
  weightAdjustmentListHiddenToOutput.push(newW6);
  var newW7 = hiddenToOutput[2] - (learningRateValue * (deltaNode[1] * hidden[0]));
  weightAdjustmentListHiddenToOutput.push(newW7);
  var newW8 = hiddenToOutput[3] - (learningRateValue * (deltaNode[1] * hidden[1]));
  weightAdjustmentListHiddenToOutput.push(newW8);
  printList(weightAdjustmentListHiddenToOutput,"Round 1 \t\tAdjusted w5,w6,w7 & w8 Weights");

  partialDerivativePart1Function(deltaNode,w5,w6,w7,w8);
  partialDerivativePart2Function(hidden);
  partialDerivativePartProductFunction(partialDerivativePart1,partialDerivativePart2);
  printList(partialDerivativePartProduct,"Round 1 \t\tPartial Derivative Product for h1 & h2");

  var newW1 = inputToHidden[0] - (learningRateValue * (partialDerivativePartProduct[0] * input[0]));
  weightAdjustmentListInputToHidden.push(newW1);
  var newW2 = inputToHidden[1] - (learningRateValue * (partialDerivativePartProduct[0] * input[1]));
  weightAdjustmentListInputToHidden.push(newW2);
  var newW3 = inputToHidden[2] - (learningRateValue * (partialDerivativePartProduct[1] * input[0]));
  weightAdjustmentListInputToHidden.push(newW3);
  var newW4 = inputToHidden[3] - (learningRateValue * (partialDerivativePartProduct[1] * input[1]));
  weightAdjustmentListInputToHidden.push(newW4);
  printList(weightAdjustmentListInputToHidden,"Round 1 \t\tAdjusted w1,w2,w3 & w4 Weights");

  //clear all temporary arrays so that the functions can be reused several times
  hiddenToOutput = weightAdjustmentListHiddenToOutput;
  inputToHidden = weightAdjustmentListInputToHidden;
  hidden = [];
  output = [];
  deltaRulePart1 = [];
  deltaRulePart2 = []; 
  deltaNode = []; 
  weightAdjustmentListHiddenToOutput = [];
  partialDerivativePart1 = [];
  partialDerivativePart2 = [];
  partialDerivativePartProduct = [];
  weightAdjustmentListInputToHidden = [];

  /*backpropagationOutputToHidden(output,targetOutputList,hidden,hiddenToOutput,learningRateValue,weightAdjustmentList);
  printList(weightAdjustmentList,"Round 1, Adjusted w5,w6,w7 & w8 Weights");*/
}

```

### Demonstrating Neural Network (Round 2)

```python
function demonstratingNeuralNetworkRound2() {
  layerToLayerForward(input,inputToHidden,iBias,hidden);
  layerToLayerForward(hidden,hiddenToOutput,hBias,output);
  printList(output,"Round 2 \t\tOutput Layer Neurons");

  console.log(" ");
  console.log("Round 2 \t\tOutput Total Error:");
  console.log("\t\t\t" + totalErrorFunction(output,targetOutputList) + " <- Note though the total error is still significant, it has decreased by several tenths. Thousands of iterations will decrease this Total Error even further.");

  deltaRulePart1Function(output,targetOutputList);
  deltaRulePart2Function(output);
  deltaNodeFunction(deltaRulePart1,deltaRulePart2);

  var newW5 = hiddenToOutput[0] - (learningRateValue * (deltaNode[0] * hidden[0]));
  weightAdjustmentListHiddenToOutput.push(newW5);
  var newW6 = hiddenToOutput[1] - (learningRateValue * (deltaNode[0] * hidden[1]));
  weightAdjustmentListHiddenToOutput.push(newW6);
  var newW7 = hiddenToOutput[2] - (learningRateValue * (deltaNode[1] * hidden[0]));
  weightAdjustmentListHiddenToOutput.push(newW7);
  var newW8 = hiddenToOutput[3] - (learningRateValue * (deltaNode[1] * hidden[1]));
  weightAdjustmentListHiddenToOutput.push(newW8);

  partialDerivativePart1Function(deltaNode,w5,w6,w7,w8);
  partialDerivativePart2Function(hidden);
  partialDerivativePartProductFunction(partialDerivativePart1,partialDerivativePart2);

  var newW1 = inputToHidden[0] - (learningRateValue * (partialDerivativePartProduct[0] * input[0]));
  weightAdjustmentListInputToHidden.push(newW1);
  var newW2 = inputToHidden[1] - (learningRateValue * (partialDerivativePartProduct[0] * input[1]));
  weightAdjustmentListInputToHidden.push(newW2);
  var newW3 = inputToHidden[2] - (learningRateValue * (partialDerivativePartProduct[1] * input[0]));
  weightAdjustmentListInputToHidden.push(newW3);
  var newW4 = inputToHidden[3] - (learningRateValue * (partialDerivativePartProduct[1] * input[1]));
  weightAdjustmentListInputToHidden.push(newW4);

  hiddenToOutput = weightAdjustmentListHiddenToOutput;
  inputToHidden = weightAdjustmentListInputToHidden;
  hidden = [];
  output = [];
  deltaRulePart1 = [];
  deltaRulePart2 = []; 
  deltaNode = []; 
  weightAdjustmentListHiddenToOutput = [];
  partialDerivativePart1 = [];
  partialDerivativePart2 = [];
  partialDerivativePartProduct = [];
  weightAdjustmentListInputToHidden = [];

  console.log("****************************************************************************************************************")
}

```

### Running Neural Network

```python
function runningNeuralNetwork() {
  var count = 3;
  while (count <= 10000) {
    layerToLayerForward(input,inputToHidden,iBias,hidden);
    layerToLayerForward(hidden,hiddenToOutput,hBias,output);
    printList(output,"Round " + count + " \t\tOutput Layer Neurons");
    console.log(" ");
    console.log(" \t\t\tOutput Total Error:");
    console.log("\t\t\t" + totalErrorFunction(output,targetOutputList));

    deltaRulePart1Function(output,targetOutputList);
    deltaRulePart2Function(output);
    deltaNodeFunction(deltaRulePart1,deltaRulePart2);

    var newW5 = hiddenToOutput[0] - (learningRateValue * (deltaNode[0] * hidden[0]));
    weightAdjustmentListHiddenToOutput.push(newW5);
    var newW6 = hiddenToOutput[1] - (learningRateValue * (deltaNode[0] * hidden[1]));
    weightAdjustmentListHiddenToOutput.push(newW6);
    var newW7 = hiddenToOutput[2] - (learningRateValue * (deltaNode[1] * hidden[0]));
    weightAdjustmentListHiddenToOutput.push(newW7);
    var newW8 = hiddenToOutput[3] - (learningRateValue * (deltaNode[1] * hidden[1]));
    weightAdjustmentListHiddenToOutput.push(newW8);

    partialDerivativePart1Function(deltaNode,w5,w6,w7,w8);
    partialDerivativePart2Function(hidden);
    partialDerivativePartProductFunction(partialDerivativePart1,partialDerivativePart2);

    var newW1 = inputToHidden[0] - (learningRateValue * (partialDerivativePartProduct[0] * input[0]));
    weightAdjustmentListInputToHidden.push(newW1);
    var newW2 = inputToHidden[1] - (learningRateValue * (partialDerivativePartProduct[0] * input[1]));
    weightAdjustmentListInputToHidden.push(newW2);
    var newW3 = inputToHidden[2] - (learningRateValue * (partialDerivativePartProduct[1] * input[0]));
    weightAdjustmentListInputToHidden.push(newW3);
    var newW4 = inputToHidden[3] - (learningRateValue * (partialDerivativePartProduct[1] * input[1]));
    weightAdjustmentListInputToHidden.push(newW4);

    hiddenToOutput = weightAdjustmentListHiddenToOutput;
    inputToHidden = weightAdjustmentListInputToHidden;
    hidden = [];
    output = [];
    deltaRulePart1 = [];
    deltaRulePart2 = []; 
    deltaNode = []; 
    weightAdjustmentListHiddenToOutput = [];
    partialDerivativePart1 = [];
    partialDerivativePart2 = [];
    partialDerivativePartProduct = [];
    weightAdjustmentListInputToHidden = [];

    count++;
  }
}
```

### Layer to Layer Forward 

```python
function layerToLayerForward(currentLayerList, weightList, biasValue, nextLayerList) {

  neuronCount = currentLayerList.length;

  var sum1 = 0;
  var finalSum1 = 0;

  for (var a = 0 ; a < currentLayerList.length ; a++) {
    if (a < (neuronCount-1) )
    {
      for (var b = 0 ; b < neuronCount ; b++)
      {
        sum1 += currentLayerList[a] * weightList[b];
      }
    }
    else
    {
      for (var c = neuronCount ; c < (2*neuronCount) ; c++)
      {
        sum1 += currentLayerList[a] * weightList[c];
      }
    }

  sum1 += biasValue * 1;
  finalSum1 = logisticalFunction(sum1);
  nextLayerList.push(finalSum1);

  sum1 = 0;
  finalSum1 = 0;
  }
}
```

### Logistical Function

```python
function logisticalFunction(netInput) {
  netInput *= -1;
  var functionResult = 1 / ( 1 + Math.exp(netInput) );
  return functionResult;
}
```

### Total Error Function

```python
function totalErrorFunction(currentOutputList,targetOutputListParameter) {
  var tempTotalError = 0;
  var totalErrorList = [];
  var finalTotalError = 0;

  for (var e = 0 ; e < currentOutputList.length ; e++) {
    tempTotalError = 0.5 * Math.pow((targetOutputListParameter[e] - currentOutputList[e]),2);
    totalErrorList.push(tempTotalError);
  };

  for (var f = 0 ; f < totalErrorList.length ; f++)
    finalTotalError += totalErrorList[f];;

  return finalTotalError;
}
```

### Delta Rule (Part 1)

```python
function deltaRulePart1Function(currentOutputList,targetOutputListParameter) {
  for (var g = 0 ; g < currentOutputList.length ; g++) {
    deltaRulePart1.push( -1 * (targetOutputListParameter[g] - currentOutputList[g]) );
  }
}
```

### Delta Rule (Part 2)

```python
function deltaRulePart2Function(currentOutputList) {
  for (var h = 0 ; h < currentOutputList.length ; h++) {
    deltaRulePart2.push(currentOutputList[h] * (1 - currentOutputList[h]));
  }
}
```

### Delta Node Function

```python
function deltaNodeFunction(deltaRulePart1List,deltaRulePart2List) {
  for (var i = 0 ; i < deltaRulePart1List.length ; i++) {
    deltaNode.push(deltaRulePart1List[i] * deltaRulePart2List[i]);
  }
}
```

### Partial Derivative (Part 1)

```python
function partialDerivativePart1Function (deltaNodeList,weight5,weight6,weight7,weight8) {

  var tempPartialDerivativePart1a = deltaNode[0] * weight5;
  var tempPartialDerivativePart1b = deltaNode[1] * weight6;
  var totalTemp = tempPartialDerivativePart1a + tempPartialDerivativePart1b;
  partialDerivativePart1.push(totalTemp);

  tempPartialDerivativePart1a = deltaNode[0] * weight7;
  tempPartialDerivativePart1b = deltaNode[1] * weight8;
  totalTemp = tempPartialDerivativePart1a + tempPartialDerivativePart1b;
  partialDerivativePart1.push(totalTemp);

} 
```

### Partial Derivative (Part 2)

```python
function partialDerivativePart2Function (currentHiddenList) {
  partialDerivativePart2.push( currentHiddenList[0] * (1 - currentHiddenList[0]) );
  partialDerivativePart2.push( currentHiddenList[1] * (1 - currentHiddenList[1]) );
}
```

### Partial Derivative - Partial Product

```python
function partialDerivativePartProductFunction (partialDerivativePart1List,partialDerivativePart2List) {
  partialDerivativePartProduct.push( partialDerivativePart1[0] * partialDerivativePart2[0]);
  partialDerivativePartProduct.push( partialDerivativePart1[1] * partialDerivativePart2[1]);
}
```

### Print List

```python
function printList(currentLayerList,label) {
  console.log(" ");
  console.log(label + ":");
  for (var d = 0 ; d < currentLayerList.length ; d++)
    console.log("\t\t\t" + currentLayerList[d]);
}
```

### Backpropagation Output to Hidden

This function proves faulty and requiring of review. Should be commented out when running in console. The program runs as intended without this function.

```python
function backpropagationOutputToHidden(currentOutputList,targetOutputListParameter,currentHiddenList,hiddenToOutputWeights,learningRateParameter,newHiddenToOutputWeights) {

  var finalDeltaProduct = 0;
  var newWeightValue = 0;

  deltaRulePart1Function(currentOutputList,targetOutputListParameter);
  deltaRulePart2Function(currentOutputList);
  deltaNodeFunction(deltaRulePart1,deltaRulePart2);

  for (var j = 0 ; j < currentOutputList.length ; j++) {
    for (var k = 0 ; k < currentHiddenList ; k++) {
      finalDeltaProduct = deltaNode[j] * currentHiddenList[k];
      if (j == 0) {
        newWeightValue = hiddenToOuputWeights[k] - (learningRateParameter * finalDeltaProduct);
        newHiddenToOutputWeights.push(newWeightValue)
      }
      else {
        newWeightValue = hiddenToOuputWeights[k+2] - (learningRateParameter * finalDeltaProduct);
        newHiddenToOutputWeights.push(newWeightValue);
      }
    }
  }
}
```
