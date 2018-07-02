/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.opennlp.namecat;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class NameCategorizer {

  private final Session session;
  private final Map<Character, Integer> charMap = new HashMap<>();
  private final Map<Integer, String> labelMap = new HashMap<>();

  public NameCategorizer(InputStream vocabChars, InputStream modelZipPackage) throws IOException {

    try (BufferedReader in = new BufferedReader(new InputStreamReader(vocabChars, "UTF8"))) {
      String ch;
      int idx = 0;
      while ((ch = in.readLine()) != null) {
        charMap.put(ch.charAt(0), idx);
        idx += 1;
      }
    }

    //Path tmpModelPath = ModelUtil.writeModelToTmpDir(modelZipPackage);

    Path tmpModelPath = Paths.get("/home/blue/dev/opennlp-sandbox/tf-ner-poc/" +
        "src/main/python/namecat/namecat_model9");

    SavedModelBundle model = SavedModelBundle.load(tmpModelPath.toString(), "serve");
    session = model.session();
  }


  public String[] categorize(String[] names) {
    if (names.length == 0) {
      return new String[0];
    }

    int maxLength = Arrays.stream(names).mapToInt(String::length).max().getAsInt();

    int charIds[][] = new int[names.length][maxLength];
    int nameLengths[] = new int[names.length];

    for (int nameIndex = 0; nameIndex < names.length; nameIndex++) {
      for (int charIndex = 0; charIndex < names[nameIndex].length(); charIndex++) {
        charIds[nameIndex][charIndex] = charMap.get(names[nameIndex].charAt(charIndex));
      }
      nameLengths[nameIndex] = names[nameIndex].length();
    }

    try (Tensor<?> dropout = Tensor.create(1f, Float.class);
         Tensor<?> charTensor = Tensor.create(charIds);
         Tensor<?> nameLength = Tensor.create(nameLengths)) {
      List<Tensor<?>> result = session.runner()
          .feed("dropout_keep_prop", dropout)
          .feed("char_ids", charTensor)
          .feed("name_lengths", nameLength)
          .fetch("norm_probs", 0).run();

      Tensor<?> probTensor = result.get(0);

      // argmax and look up id

      float[][] probs = probTensor.copyTo(new float[1][2]);

      List<String> cats = new ArrayList<>();
      for (float[] prob : probs) {
        if (prob[0] > 0.5) {
          cats.add("F");
        }
        else {
          cats.add("M");
        }
      }

      return cats.toArray(new String[2]);
    }
  }

  public static void main(String[] args) throws Exception {
    NameCategorizer cat = new NameCategorizer(
        new FileInputStream("/home/blue/dev/opennlp-sandbox/tf-ner-poc/src/main/python/namecat/char_dict.txt"),
        null);

    System.out.println(cat.categorize(new String[]{"Lena Krakau"})[0]);
  }
}
