"use strict";

const LOG_ON = true; // whether to show error logging or not
const LOG_FREQUENCY = 20000; // how often to show error log

class  NeuralNetwork{
    constructor(numInputs,numHidden,numOutputs) {
        this._hidden = [];
        this._inputs = [];
        this._numInputs = numInputs;
        this._numHidden = numHidden;
        this._numOutputs = numOutputs;
        this._bias0 = new Matrix(1,this._numHidden);
        this._bias1 = new Matrix(1,this._numOutputs);
        this._weights0 = new Matrix(this._numInputs,this._numHidden);
        this._weights1 = new Matrix(this._numHidden,this._numOutputs);
        
        // error logging
        this._logCount = LOG_FREQUENCY;

        // randomise the initial weights
        this._bias0.randomWeights();
        this._bias1.randomWeights();
        this._weights0.randomWeights();
        this._weights1.randomWeights();
    }
    get bias1() {
        return this._bias1;
    }
    set bias1(bias)
    {
        this._bias1 = bias;
    }
    get bias0() {
        return this._bias0;
    }
    set bias0(bias)
    {
        this._bias0 = bias;
    }
    get inputs() {
        return this._inputs;
    }
    set inputs(inputs)
    {
        this._inputs = inputs;
    }
    get hidden() {
        return this._hidden;
    }
    set hidden(hidden)
    {
        this._hidden = hidden;
    }
    get weights0() {
        return this._weights0;
    }
    
    set weights0(weights)
    {
        this._weights0 = weights;
    }

    get weights1() {
        return this._weights1;
    }
    
    set weights1(weights)
    {
        this._weights1 = weights;
    }
    
    get logCount() {
        return this._logCount;
    }
    
    set logCount(count)
    {
        this._logCount = count;
    }
    // FEED FORWARD
    
    feedForward(inputArray) {
        // convert the input array to a matrix
        this.inputs = Matrix.convertFromArray(inputArray);
        /*
        console.log("inputs");
        console.table(inputs.data);
        */
        // find the hidden values and apply the activation function
        this.hidden = Matrix.dot(this.inputs,this.weights0);
        // apply bias
        this.hidden = Matrix.add(this.hidden, this.bias0); //applied bias
        /*
        console.log("hidden");
        console.table(hidden.data);
        */
        this.hidden = Matrix.map(this.hidden, x => sigmoid(x));
        /*
        console.log("hidden sigmoid");
        console.table(hidden.data);
        */
        // find the otuput values and apply the activation function
        let outputs = Matrix.dot(this.hidden,this.weights1);
        // apply bias 
        outputs = Matrix.add(outputs, this.bias1); //applied bias
        /*
        console.log("outputs");
        console.table(outputs.data);
        */
        outputs = Matrix.map(outputs, x => sigmoid(x));
        /*
        console.log("outputs sigmoid");
        console.table(outputs.data);
        */
        return outputs;
        
    }

    
    // TRAIN using the input and target arrays (one dimensional)
    train(inputArray,targetArray) {
    // feed the input data through the network
    let outputs = this.feedForward(inputArray);
    /*
    console.log("outputs");
    console.table(outputs.data);
    */
    // calculate the output errors (target - output)
    let targets = Matrix.convertFromArray(targetArray);
    /*
    console.log("targets");
    console.table(targets.data);
    */
    let outputErrors = Matrix.substract(targets, outputs);
    
    // error logging
    if (LOG_ON) {
        if (this.logCount == LOG_FREQUENCY) {
            console.log("error = " + outputErrors.data[0][0]);
        }
        this.logCount --;
        if(this.logCount == 0) {
            this.logCount = LOG_FREQUENCY;
        }
    }
    
    /*
    console.log("outputErrors");
    console.table(outputErrors.data);
    */
    // calculate the deltas (errors * derivative of the output)
    let outputDerivs = Matrix.map(outputs, x => sigmoid(x, true));
    let outputDeltas = Matrix.multiply(outputErrors, outputDerivs);
    /*
    console.log("outputDeltas");
    console.table(outputDeltas.data);
    */
    // calculate the hidden layer errors (deltas "dot" transpose of weights)
    let weightsIT = Matrix.transpose(this.weights1);
    let hiddenErrors = Matrix.dot(outputDeltas,weightsIT);
    /*
    console.log("hiddenErrors");
    console.table(hiddenErrors.data);
    */
    // calculate the hidden deltas (errors * derivative of hidden)
    let hiddenDerivs = Matrix.map(this.hidden, x => sigmoid(x, true));
    let hiddenDeltas = Matrix.multiply(hiddenErrors, hiddenDerivs);
    /*
    console.log("hiddenDeltas");
    console.table(hiddenDeltas.data);
    */
    
    // update the weights (add transpose of layers "dot" deltas)
    let hiddenT = Matrix.transpose(this.hidden);
    this.weights1 = Matrix.add(this.weights1, Matrix.dot(hiddenT, outputDeltas));
    let inputsT = Matrix.transpose(this.inputs);
    this.weights0 = Matrix.add(this.weights0, Matrix.dot(inputsT, hiddenDeltas));
    
    // update bias
    this.bias1 = Matrix.add(this.bias1, outputDeltas);
    this.bias0 = Matrix.add(this.bias0, hiddenDeltas);
    }

}

//sigmoid function to convert any value in a range between -1 and 1
function sigmoid(x, deriv = false) {
    if (deriv) {
        return x * (1 - x); // where x = sigmoid(x)
    }
    return 1 / (1 + Math.exp(-x));
}

// NEURAL NETWORK ip/hid/op


/***********************
    MATRIX FUNCTIONS
 ***********************/
class Matrix{
    constructor(rows,cols,data=[]){
        this._rows = rows;
        this._cols = cols;
        this._data = data;

        //initialize with zeroes if no data provided
        if(data == null || data.length == 0) {
            this._data = [];
            for(let i=0; i < this._rows; i++) {
                this._data[i]=[];
                for(let j=0; j < this._cols;j++) {
                    this._data[i][j]=0;
                }
            }
        } else {
            // check data integrity
            if(data.length != rows||data[0].length != cols) {
                throw new Error("Incorrect data dimensions!");
            }
        }
        
    }
    get rows() {
        return this._rows;
    }
    
    get cols() {
        return this._cols;
    }

    get data() {
        return this._data;
    }

    // add two matrices
    static add(m0,m1) {
        Matrix.checkDimensions(m0,m1);
        let m = new Matrix(m0.rows,m0.cols);
        for (let i = 0; i < m.rows;i++){
            for (let j=0; j< m.cols;j++){
                m.data[i][j] = m0.data[i][j] + m1.data[i][j]; 
            }
        }
        return m;
    }
    
    // check dimensions of two matrices

    static checkDimensions(m0, m1) {
        if(m0.rows != m1.rows || m0.cols != m1.cols) {
            throw new Error("Matrices are of different dimensions!");
        }
    }
    // substract two matrices
    static substract(m0,m1) {
        Matrix.checkDimensions(m0,m1);
        let m = new Matrix(m0.rows,m0.cols);
        for (let i = 0; i < m.rows;i++){
            for (let j=0; j< m.cols;j++){
                m.data[i][j] = m0.data[i][j] - m1.data[i][j]; 
            }
        }
        return m;
    }
    // multiply two matrices (not the dot product)
  
    static multiply(m0,m1) {
        Matrix.checkDimensions(m0,m1);
        let m = new Matrix(m0.rows,m0.cols);
        for (let i = 0; i < m.rows;i++){
            for (let j=0; j< m.cols;j++){
                m.data[i][j] = m0.data[i][j] * m1.data[i][j]; 
            }
        }
        return m;
    }
    // apply a function to each cell of the given matrix
    static map(m0, mFunction) {
        let m = new Matrix(m0.rows,m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = mFunction(m0.data[i][j]);
            }
        }
        return m;
    }

    // dot product of two matrices
    static dot(m0,m1){
        if (m0.cols != m1.rows) {
            throw new Error("Matrices are not \"dot\" compatible!");
        }
        let m = new Matrix(m0.rows, m1.cols);
        for (let i = 0; i < m.rows;i++){
            for (let j=0; j< m.cols;j++){
                let sum = 0;
                for (let k = 0; k < m0.cols; k++) {
                    sum += m0.data[i][k] * m1.data[k][j];
                }
                 m.data[i][j] = sum;
            }
        }
        return m;
    }

    // find the transpose of a given matrix
    static transpose(m0){
        let m = new Matrix(m0.cols,m0.rows);
        for (let i = 0; i < m0.rows;i++){
            for (let j=0; j< m0.cols;j++){
                m.data[j][i] = m0.data[i][j];
            }
        }
        return m;            
    }

    // convert array to one-rowed matrix

    static convertFromArray(arr){
        return new Matrix(1,arr.length,[arr]);
    }

    // apply random weights between -1 and 1
    randomWeights()
    {
        for(let i=0; i< this.rows; i++) {
            for(let j = 0; j< this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1
            }
        }
    }
}