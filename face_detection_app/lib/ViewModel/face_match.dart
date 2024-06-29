import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../Model/db.dart';
import 'face_register.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;

  final dbHelper = DatabaseHelper();
  Map<String, Recognition> registered = {};

  @override
  Recognizer({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    init();
  }

  Future<void> init() async {
    await loadModel();
    await initDB();
  }

  initDB() async {
    await dbHelper.init();
    await loadRegisteredFaces();
  }
  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/model/facenet.tflite',
          options: _interpreterOptions);
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  Future<void> loadRegisteredFaces() async {
    registered.clear();
    final allRows = await dbHelper.queryAllRows();

    // Future.forEach ile tüm işlemleri sırayla bekleyecek şekilde yap
    await Future.forEach(allRows, (row) async {
      print(row[DatabaseHelper.columnFirstName]);
      String name = row[DatabaseHelper.columnFirstName];
      String surname = row[DatabaseHelper.columnLastName];
      String fullName = '$name $surname';

      String embedding0 = row[DatabaseHelper.columnEmbedding];
      String cleanembedding = embedding0.substring(1, embedding0.length - 1);

      List<double> doubleList = cleanembedding
          .split(',')
          .map((e) => double.parse(e))
          .toList()
          .cast<double>();
      String imagePath = row[DatabaseHelper.columnImagePath];
      Recognition recognition = Recognition(
          row[DatabaseHelper.columnFirstName], Rect.zero, doubleList, 0,imagePath);
      registered.putIfAbsent(fullName, () => recognition);
    });
  }

  void registerFaceInDB(String number, String name, String surname,
      List<double> embedding,Uint8List face) async {
    String imagePath = await saveImageAndGetPath(face);
    Map<String, dynamic> row = {
      DatabaseHelper.columnStdNumber: number,
      DatabaseHelper.columnFirstName: name,
      DatabaseHelper.columnLastName: surname,
      DatabaseHelper.columnEmbedding: embedding.join(","),
      DatabaseHelper.columnImagePath: imagePath
    };
    final id = await dbHelper.insert(row);
    print('inserted row id: $id');
    loadRegisteredFaces();
  }
  Future<String> saveImageAndGetPath(Uint8List imageBytes) async {
    // Get the temporary directory path
    Directory tempDir = await getTemporaryDirectory();
    String tempPath = tempDir.path;

    // Generate a unique file name using timestamp
    String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
    String imagePath = '$tempPath/$timestamp.jpg';

    // Save the image to the temporary directory
    await File(imagePath).writeAsBytes(imageBytes);

    return imagePath;
  }

  List<dynamic> imageToArray(img.Image inputImage) {
    img.Image resizedImage =
        img.copyResize(inputImage, width: 160, height: 160);
    List<double> flattenedList = resizedImage.data!
        .expand((channel) => [channel.r, channel.g, channel.b])
        .map((value) => value.toDouble())
        .toList();
    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    int height = 160;
    int width = 160;
    Float32List reshapedArray = Float32List(1 * height * width * channels);
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] =
              (float32Array[c * height * width + h * width + w] - 127.5) /
                  127.5;
        }
      }
    }
    return reshapedArray.reshape([1, 160, 160, 3]);
  }
  // List<dynamic> imageTOARRAY (img.Image inputImage){}

  Future<Recognition> recognize(img.Image image, Rect location) async {
    var input = imageToArray(image);
    
    List output = List.filled(1 * 512, 0).reshape([1, 512]);

    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    final run = DateTime.now().millisecondsSinceEpoch - runs;
    String stringOutput = output.first.toString();
    String cleanOutput = stringOutput.substring(1, stringOutput.length - 1);

    List<double> doubleListOutput = cleanOutput
        .split(',')
        .map((e) => double.parse(e))
        .toList()
        .cast<double>();
    Pair pair = findNearest(doubleListOutput);
    print("mesafe= ${pair.distance}");

    String imagePath = await saveImageAndGetPath(img.encodeJpg(image));

    return Recognition(pair.number, location, doubleListOutput, pair.distance, imagePath);
  }


  double cosineSimilarity(List<double> vector1, List<double> vector2) {
    // Hesaplamak için vektörlerin boyutlarını kontrol etmek önemlidir.
    if (vector1.length != vector2.length ||
        vector1.isEmpty ||
        vector2.isEmpty) {
      throw ArgumentError("Vektörlerin boyutları eşit ve boş olmamalıdır.");
    }

    // İki vektörün iç çarpımını hesapla
    double dotProduct = 0;
    for (int i = 0; i < vector1.length; i++) {
      dotProduct += vector1[i] * vector2[i];
    }

    // İki vektörün normlarını hesapla
    double norm1 = vectorNorm(vector1);
    double norm2 = vectorNorm(vector2);

    // Kosinüs benzerliği hesapla
    double similarity = dotProduct / (norm1 * norm2);

    return similarity;
  }

  double vectorNorm(List<double> vector) {
    // Vektörün normunu hesapla (Euclidean norm)
    double sum = 0;
    for (int i = 0; i < vector.length; i++) {
      sum += vector[i] * vector[i];
    }
    return sqrt(sum);
  }

  /*findNearest(List<double> embedding) {
    Pair pair = Pair("Yüz Tanınmıyor", -5);

    for (MapEntry<String, Recognition> item in registered.entries) {
      final String number = item.key;
      List<double> knownEmb = item.value.embedding;

      // Kosinüs benzerliğini hesapla
      double similarity = cosineSimilarity(embedding, knownEmb);

      if (pair.distance == -5 || similarity > pair.distance) {
        pair.distance = similarity;
        pair.number = number;
      }
    }

    return pair;
  }*/

  //öklid mesafesi ile benzelik hesaplamak için kullanılan fonksiyon
  /*findNearest(List<double> embedding) {
    Pair pair = Pair("Yüz Tanınmıyor", -5);

    for (MapEntry<String, Recognition> item in registered.entries) {
      if (registered.entries.isEmpty) {
        print('empty');
      }
      print(registered.entries);
      final String number = item.key;
      print(number);
      List<double> knownEmb = item.value.embedding;
      double distance = 0;
      for (int i = 0; i < embedding.length; i++) {
        double diff = embedding[i] - knownEmb[i];
        distance += diff * diff;
      }
      distance = sqrt(distance);
      if (pair.distance == -5 || distance < pair.distance) {
        pair.distance = distance;
        pair.number = number;
      }
    }
    return pair;
  }
*/
/*findNearest(List<double> embedding) {
    Pair pair = Pair("Yüz Tanınmıyor", -5);

    for (MapEntry<String, Recognition> item in registered.entries) {
      if (registered.entries.isEmpty) {
        print('empty');
      }
      print(registered.entries);
      final String number = item.key;
      print(number);
      List<double> knownEmb = item.value.embedding;

      double distance = 0;
      for (int i = 0; i < embedding.length; i++) {
        double diff = embedding[i] - knownEmb[i];
        distance += diff * diff;
      }
      distance = sqrt(distance);
      if (pair.distance == -5 || distance < pair.distance) {
        pair.distance = distance;
        pair.number = number;
      }
    }
    return pair;
  } */
  findNearest(List<double> embedding) {
    Pair pair = Pair("Yüz Tanınmıyor %",
        double.infinity); // Eşleşme bulunamazsa mesafe 5 olarak ayarlanır

    for (MapEntry<String, Recognition> item in registered.entries) {
      if (registered.entries.isEmpty) {
        print('empty');
        continue; // Kayıtlı giriş yoksa döngüyü atlayın
      }

      final String number = item.key;
      List<double> knownEmb = item.value.embedding;

      double distance =
          euclideanDistance(embedding, knownEmb); // Öklid mesafesini hesaplayın

      if (distance < pair.distance) {
        // Daha yakın bir eşleşme bulunursa güncelleyin
        pair = Pair(number, distance);
      }
    }

    return pair;
  }

  double euclideanDistance(List<double> embedding, List<double> knownEmb) {
    double distance = 0;
    for (int i = 0; i < embedding.length; i++) {
      double diff = embedding[i] - knownEmb[i];
      distance += diff * diff;
    }
    distance = sqrt(distance);
    return distance;
  }

  void close() {
    interpreter.close();
  }
}

class Pair {
  String number;
  double distance;
  Pair(this.number, this.distance);
}
