package org.deeplearning4j.examples;

import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.eval.ROC;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;

import java.io.File;
import java.util.concurrent.TimeUnit;

public class HFPredictor {
    public static void main(String[] args) throws Exception{


        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader trainR = new CSVRecordReader(numLinesToSkip, delimiter);
        trainR.initialize(new FileSplit(new File("heart_failure_clinical_records_dataset_train.csv")));

        RecordReader testR = new CSVRecordReader(numLinesToSkip, delimiter);
        testR.initialize(new FileSplit(new File("heart_failure_clinical_records_dataset_test.csv")));


        int labelIndex = 12;
        int numClasses = 2;
        int TrainbatchSize = 219;
        int TestbatchSize = 80;
        DataSetIterator TrainIterator = new RecordReaderDataSetIterator(trainR, TrainbatchSize, labelIndex, numClasses);
        DataSetIterator TestIterator = new RecordReaderDataSetIterator(testR, TestbatchSize, labelIndex, numClasses);


        // normalization

        DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(TrainIterator);
        TrainIterator.setPreProcessor(dataNormalization);
        TestIterator.setPreProcessor(dataNormalization);

        /* DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.7);  //Use 70% of data for training */

        // DataSet trainingData = TrainIterator.next();


//Neural Network

        long seed = 6;


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(labelIndex).nOut(4)
                .activation(Activation.TANH)
                .build())
            .layer(new DenseLayer.Builder().nIn(4).nOut(14)
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder().nIn(14).nOut(4)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(4).nOut(numClasses).build())
            .build();



        //Training with early stopping

        String  tempDir= System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        File dirFile  = new File(exampleDirectory);
        dirFile.mkdir();

        LocalFileModelSaver saver  = new LocalFileModelSaver(exampleDirectory);

        EarlyStoppingConfiguration esConf  = new EarlyStoppingConfiguration.Builder()
            .epochTerminationConditions(new MaxEpochsTerminationCondition(1000))
            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(5, TimeUnit.MINUTES))
            .scoreCalculator(new DataSetLossCalculator(TestIterator, true))
            .evaluateEveryNEpochs(1)
            .modelSaver(saver)
            .build();

        // KFoldIterator cross= new KFoldIterator(trainingData); //K=10

        EarlyStoppingTrainer trainer  = new EarlyStoppingTrainer(esConf,conf,TrainIterator);
        EarlyStoppingResult<MultiLayerNetwork> result=trainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        MultiLayerNetwork model =result.getBestModel();
        model.init();

       /* model.setListeners(new ScoreIterationListener(20));

        model.fit(TrainIterator,10);  */

        /* Evaluation eval = new Evaluation(2);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output); */

        //Evaluation

        Evaluation eval = model.evaluate(TestIterator);
        System.out.println(eval.stats());

        System.out.println(model.evaluateROC(TestIterator, 100).calculateAUC());

    }
}
