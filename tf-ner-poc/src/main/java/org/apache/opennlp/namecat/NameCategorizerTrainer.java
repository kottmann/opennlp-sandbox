package org.apache.opennlp.namecat;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.opennlp.tf.Saver;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class NameCategorizerTrainer {


  private static Tensor createCharIdTensor(int batchSize, int batchIndex,
                                         Map<Character, Integer> charMap, List<String> names) {
    int begin = batchSize * batchIndex;
    int end = Math.min(begin + batchSize, names.size());

    int maxLength = names.subList(begin, end).stream()
        .mapToInt(String::length).max().getAsInt();

    int[][] charIds = new int[end - begin][maxLength];

    for (int i = begin; i < end; i++) {
      String name = names.get(i);

      for (int charIdx = 0; charIdx < name.length(); charIdx++) {
        charIds[i - begin][charIdx] = charMap.get(name.charAt(charIdx));
      }
    }

    return Tensor.create(charIds);
  }

  private static Tensor createNameLengthTensor(int batchSize, int batchIndex, List<String> names) {
    int begin = batchSize * batchIndex;
    int end = Math.min(begin + batchSize, names.size());

    int[] length = new int[end - begin];

    for (int i = begin; i < end; i++) {
      length[i - begin] = names.get(i).length();
    }

    return Tensor.create(length);
  }

  private static Tensor createCategoryTensor(int batchSize, int batchIndex,
                                             Map<String, Integer> categoryMap,  List<String> labels) {
    int begin = batchSize * batchIndex;
    int end = Math.min(begin + batchSize, labels.size());

    int[] categories = new int[end - begin];

    for (int i = begin; i < end; i++) {
      categories[i - begin] = categoryMap.get(labels.get(i));
    }

    return Tensor.create(categories);
  }

  public static void main(String[] args) throws Exception {

    // TODO: the python code could be run as part of the java training to create the right
    // graph based on the input chars, and input classes
    // Is there no other way to build a generic graph ?!?!?!?

    // Is it possible to create the graph partially in java ?!?!?
    // LSTM ops are missing currently (but could be added)

    List<String> labels = new ArrayList<>();
    List<String> names = new ArrayList<>();

    for (String line : Files.readAllLines(Paths.get("/home/blue/dev/d2/names.txt/train.txt"))) {
      String[] parts = line.split("\t");
      labels.add(parts[0]);
      names.add(parts[1]);
    }

    Map<Character, Integer> charMap = new HashMap<>();
    for (String name : names) {
      for (int i = 0; i < name.length(); i++) {
        charMap.putIfAbsent(name.charAt(i), charMap.size());
      }
    }

    Map<String, Integer> categoryMap = new HashMap<>();
    for (String category : labels) {
      categoryMap.putIfAbsent(category, categoryMap.size());
    }

    // TODO: Store the maps in the model after training

    // TODO: Try to use input_map to replace the hard coded variables, with new ones



    SavedModelBundle model = SavedModelBundle.load(
        "/home/blue/dev/opennlp-sandbox/tf-ner-poc/src/main/python/namecat/namecat_graph", "train" );

    Session sess = model.session();

    int batchSize = 20;
    int batchCount = names.size() / batchSize;

    for (int epoch = 0; epoch < 10; epoch++) { // Iteration over epochs

      System.out.printf("Epoch %d\n", epoch);

      for (int batchIndex = 6900; batchIndex < batchCount; batchIndex++) {

        Tensor dropOutKeepProp = Tensor.create(new float[] {0.7f});
        Tensor charIds = createCharIdTensor(batchSize, batchIndex, charMap, names);
        Tensor lengths = createNameLengthTensor(batchSize, batchIndex, names);
        Tensor category = createCategoryTensor(batchSize, batchIndex, categoryMap, labels);

        sess.runner()
            .feed("dropout_keep_prop", dropOutKeepProp)
            .feed("char_ids", charIds)
            .feed("name_lengths", lengths)
            .feed("category", category)
            .addTarget("train").run();

        System.out.printf("batch index %d\n", batchIndex);
      }
    }

    Saver saver = new Saver(false);
    saver.save(sess, Paths.get("check-dir"));

    // TODO: Implement SavedModelBuilder
  }
}
