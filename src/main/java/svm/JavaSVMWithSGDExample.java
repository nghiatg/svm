package svm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

// $example on$
import scala.Tuple2;
// $example off$

/**
 * Example for SVMWithSGD.
 */
public class JavaSVMWithSGDExample {
	public static void main(String[] args) throws IOException {
		SparkConf conf = new SparkConf().setAppName("JavaSVMWithSGDExample").setMaster("local[2]");
		SparkContext sc = new SparkContext(conf);
		sc.setLogLevel("ERROR");
		// $example on$
		String path = "E:\\study\\text\\bias\\svm_data";
		// String path =
		// "C:\\Users\\VCCORP\\Desktop\\label\\spark_sample\\libsvm_data.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();
		JavaRDD<LabeledPoint> realTest = MLUtils.loadLibSVMFile(sc, "E:\\study\\text\\bias\\fromtest").toJavaRDD();

		// Split initial RDD into two... [60% training data, 40% testing data].
		JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
		training.cache();
		JavaRDD<LabeledPoint> test = data.subtract(training);
		
		// Run training algorithm to build the model.
		int numIterations = 1000;
		SVMModel model = SVMWithSGD.train(data.rdd(), numIterations);

		// Clear the default threshold.
		 model.clearThreshold();
		model.setThreshold(-0.04);

		// Compute raw scores on the test set.
		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = realTest
				.map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));

		System.out.println("Area under ROC = " + metrics.areaUnderROC());

//		ArrayList<Integer> fc = new ArrayList<Integer>();
		for (int i = 0 ; i < scoreAndLabels.collect().size() ; ++i) {
			Tuple2<Object, Object> tuple = scoreAndLabels.collect().get(i);
//			if(tuple._1.equals(tuple._2)) {
//				fc.add(i);
//			}
			System.out.println(tuple._1 + "\t" + tuple._2 + "\t" + tuple._1.equals(tuple._2));
		}
//		filter2(fc);

		// // Get evaluation metrics.
		

//		for (double i = -1.0; i < 1.0; i += 0.01) {
//			model.setThreshold(i);
//			JavaRDD<Tuple2<Object, Object>> scoreAndLabelsinside = realTest
//					.map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
//			BinaryClassificationMetrics metricsinside = new BinaryClassificationMetrics(
//					JavaRDD.toRDD(scoreAndLabelsinside));
//			System.out.println(i + "\t-->Area under ROC = " + metricsinside.areaUnderROC());
//		}

		
		// // Save and load model
		// model.save(sc, "target/tmp/javaSVMWithSGDModel");
		// SVMModel sameModel = SVMModel.load(sc, "target/tmp/javaSVMWithSGDModel");

		sc.stop();
	}

	 public static void filter(ArrayList<Integer> fc) throws IOException{
		 BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\VCCORP\\Desktop\\label\\bias\\url0"));
		 BufferedReader br2 = new BufferedReader(new FileReader("C:\\Users\\VCCORP\\Desktop\\label\\bias\\url1"));
		 PrintWriter pr = new PrintWriter("C:\\Users\\VCCORP\\Desktop\\label\\bias\\filteredURL");
		 String line = br.readLine();
		 int count = 0 ; 
		 while(line != null) {
			 if(fc.contains(count)) {
				 pr.println(line);
			 }
			 count++;
			 line = br.readLine();
		 }		 
		 line = br2.readLine();
		 while(line != null) {
			 if(fc.contains(count)) {
				 pr.println(line);
			 }
			 count++;
			 line = br2.readLine();
		 }
		 br.close();
		 br2.close();
		 pr.close();
	 }
	 
	 public static void filter2(ArrayList<Integer> fc) throws IOException {
		 BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\VCCORP\\Desktop\\label\\bias\\filteredURL"));
		 PrintWriter pr = new PrintWriter("C:\\Users\\VCCORP\\Desktop\\label\\bias\\filteredURL2");
		 String line = br.readLine();
		 int count = 0;
		 while(line != null){
			 if(fc.contains(count)) {
				 pr.println(line);
			 }
			 count++;
			 line = br.readLine();
		 }
		 br.close();
		 pr.close();
	 }
}