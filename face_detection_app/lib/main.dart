import 'package:flutter/material.dart';
import 'View/face_match_view.dart';
import 'View/facedetector.dart';
import 'View/home_view.dart';
import 'View/login_page.dart';
import 'View/register_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    MaterialApp(
      color: Colors.amber,
      theme: ThemeData(primarySwatch: Colors.amber),
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
      routes: {
        '/login': (context) => const LoginPage(),
        '/register': (context) => const RegisterPage(),
        '/homeScreen': (context) => const HomeScreen(),
        '/fromGalley': (context) => const RecognitionScreen(),
        '/fromLiveCamera': (context) => const FaceDetectorView(),
      },
    ),
  );
}
