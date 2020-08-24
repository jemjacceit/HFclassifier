package org.deeplearning4j.examples;

import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Properties;

public class DataSourceCustom implements DataSource {
    DataSet trainig = new DataSet();
    DataSet test = new DataSet();
    @Override
    public void configure(Properties properties) {

    }
    public void settrainData(DataSet data) {
        this.trainig = data;
    }
    public void setTestData(DataSet data) {
        this.test = data;
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
        return this.trainig.getClass();
    }
}
