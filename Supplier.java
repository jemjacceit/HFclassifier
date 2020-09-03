package org.nd4j.linalg.function;

/**
 * A supplier of results with no input arguments
 *
 * @param  Type of result
 */
public interface Supplier {

    /**
     * @return Result
     */
    <T> T get();

}
