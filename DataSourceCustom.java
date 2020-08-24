package org.deeplearning4j.examples;

import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Properties;

public class DataSourceCustom implements DataSource {
    DataSet trainig;
    DataSet test;
    DataSetIterator TrainIterator;
    DataSetIterator TestIterator;


    @Override
    public void configure(Properties properties) {

    }
    public void settrainData(DataSet data) {
        this.trainig = data;
    }
    public void setTestData(DataSet data) {
        this.test = data;
    }

    public void setTrainIterator(DataSetIterator trainIterator) {
        TrainIterator = trainIterator;
    }

    public void setTestIterator(DataSetIterator testIterator) {
        TestIterator = testIterator;
    }

    @Override
    public Object trainData() {
        return this.trainig;
    }

    @Override
    public Object testData() {
        return this.test;
    }

    @Override
    public Class<?> getDataType() {
        return this.TrainIterator.getClass();
    }
}
