# Todo

### Implement CNN
- Implement Flatten layer
- Implement Conv2d and MaxPool2d layers
- Implement Batch Normalization and Dropout Layer

### Improve SGD
- Use mini batch SGD instead of full batch 
- Implement Nesterov Momentum (NAG)
- Implement AdaGrad
- Implement RMSProp
- Implement Adam

### Refactor
- Remove the need to create instances of `Function`
  - Each `Function` should be a singleton object (namespace)
  - Each `Function` should have a static `apply` and `apply_derivative`
  - Allow for currying of the Function
  - A `Layer` will be defined as a `Function` with state (parameters)