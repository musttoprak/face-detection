import 'dart:convert';
import 'dart:ui';

class Recognition {
  String number;
  Rect location;
  late List<double> embedding;
  double distance;
  String imagePath;

  Recognition(this.number, this.location, dynamic embedding, this.distance,this.imagePath) {
    if (embedding is List) {
      this.embedding = List.castFrom<dynamic, double>(embedding);
    } else {
      // Eğer embedding JSON formatında Firestore'dan gelmişse
      this.embedding = List<double>.from(json.decode(embedding));
    }
  }
}
