package org.deeplearning4j.examples;




import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.data.MnistDataProvider;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;


public class HFPredictor {
    public static void main(String[] args) throws Exception {

        Schema schema = new Schema.Builder()
            .addColumnsDouble("Parameter%d", 0, 11)
            .addColumnCategorical("Death_event", "0", "1")
            .build();


        TransformProcess transformProcess = new TransformProcess.Builder(schema)
            .removeAllColumnsExceptFor("Parameter4", "Parameter7", "Death_event")
            .renameColumn("Parameter4", "Ejection fraction")
            .renameColumn("Parameter7", "Serum creatinine")
            .categoricalToInteger("Death_event")
            .build();


        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader reader = new CSVRecordReader(numLinesToSkip, delimiter);
        reader.initialize(new FileSplit(new File("heart_failure_clinical_records_dataset.csv")));
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(reader, transformProcess);


        int labelIndex = 2;
        int numClasses = 2;
        int batchSize = 299;

        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.7);  //Use 70% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);

        ContinuousParameterSpace learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);
        IntegerParameterSpace layerSizeHyperparam = new IntegerParameterSpace(2, 100);

        EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(org.deeplearning4j.eval.Evaluation.Metric.ACCURACY);

        MaxTimeCondition terminationConditions = new MaxTimeCondition(5, TimeUnit.MINUTES);

        String baseSaveDirectory = "arbiterExample/";
        File f = new File(baseSaveDirectory);
        if (f.exists()) {f.delete();}
        f.mkdir();
        FileModelSaver modelSaver = new FileModelSaver(baseSaveDirectory);


        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
            .weightInit(WeightInit.XAVIER)
            .l2(0.0001)
            //Learning rate hyperparameter: search over different values, applied to all models
            .updater(new SgdSpace(learningRateHyperparam))
            .addLayer(new DenseLayerSpace.Builder()
                //Fixed values for this layer:
                .nIn(2)
                .activation(Activation.LEAKYRELU)
                //One hyperparameter to infer: layer size
                .nOut(layerSizeHyperparam)
                .build())
            .addLayer(new OutputLayerSpace.Builder()
                .nOut(2)
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build())
            .build();

        MnistDataProvider dataProvider = new MnistDataProvider(1, batchSize); //da rimuovere, solo prova

        RandomSearchGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);
        DataSourceCustom dt = new DataSourceCustom();
        dt.settrainData(trainingData);

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
            .candidateGenerator(candidateGenerator)
            //.dataProvider(dataProvider) //da rimuovere, solo prova
            .dataSource(dt.getClass(),null) //...non riesco a trasformare il DataSet trainingSet nel primo parametro(DataSource)
            .scoreFunction(scoreFunction)
            .modelSaver(modelSaver)
            .terminationConditions(terminationConditions)
            .build();

        LocalOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());

        runner.execute();

        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        //MultiLayerNetwork bestModel = bestResult.;


        System.out.println("\n\nConfiguration of best model:\n");
        //System.out.println(bestModel.getLayerWiseConfigurations().toJson());



        String s = "Best score: " + runner.bestScore() + "\n" + "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" + "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);



        String path= "arbiterExample/34";

        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(path);


        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        DataSetIterator kFoldIterator = new KFoldIterator(trainingData);
        model.fit(kFoldIterator, 100);

        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);



    }
}
