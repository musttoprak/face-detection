import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';
import 'package:path_provider/path_provider.dart';

class DatabaseHelper {
  static const _databaseName = "MyDatabase.db";
  static const _databaseVersion = 2;

  static const table = 'my_table2';

  static const columnId = 'id';
  static const columnStdNumber = 'StdNumber';
  static const columnFirstName = 'FirstName';
  static const columnLastName = 'LastName';
  static const columnEmbedding = 'Embedding';
  static const columnImagePath = 'image_path';

  late Database _db;

  Future<void> init() async {
    final documentsDirectory = await getApplicationDocumentsDirectory();
    final path = p.join(documentsDirectory.path, _databaseName);
    _db = await openDatabase(
      path,
      version: _databaseVersion,
      onCreate: _onCreate,
    );
  }

  Future _onCreate(Database db, int version) async {
    await db.execute('''
          CREATE TABLE $table (
            $columnId INTEGER PRIMARY KEY AUTOINCREMENT,
            $columnStdNumber TEXT NOT NULL,
            $columnFirstName TEXT NOT NULL,
            $columnLastName TEXT NOT NULL,
            $columnEmbedding TEXT NOT NULL,
            $columnImagePath TEXT NOT NULL
          )
          ''');
  }

  Future<int> insert(Map<String, dynamic> row) async {
    return await _db.insert(table, row);
  }

  Future<List<Map<String, dynamic>>> queryAllRows() async {
    return await _db.query(table);
  }

  Future<int> queryRowCount() async {
    final results = await _db.rawQuery('SELECT COUNT(*) FROM $table');
    return Sqflite.firstIntValue(results) ?? 0;
  }

  Future<int> update(Map<String, dynamic> row) async {
    int id = row[columnId];
    return await _db.update(
      table,
      row,
      where: '$columnId = ?',
      whereArgs: [id],
    );
  }

  Future<int> delete(int id) async {
    return await _db.delete(
      table,
      where: '$columnId = ?',
      whereArgs: [id],
    );
  }

  Future<int> deleteAllRow() async {
    return await _db.delete(table);
  }

  Future<void> dropTable() async {
    return await _db.execute('DROP TABLE IF EXISTS $table');
  }

  Future<List<Map<String, dynamic>>> getAllFaces() async {
    return await _db.query(table, columns: [
      columnId,
      columnStdNumber,
      columnFirstName,
      columnLastName,
      columnEmbedding
    ]);
  }

  Future<int> queryRowCountWhere(String column, String value) async {
    final results = await _db
        .rawQuery('SELECT COUNT(*) FROM $table WHERE $column = ?', [value]);
    return Sqflite.firstIntValue(results) ?? 0;
  }

  void _toast(String message) {
    Fluttertoast.showToast(
      msg: message,
      toastLength: Toast.LENGTH_SHORT,
      gravity: ToastGravity.CENTER,
      timeInSecForIosWeb: 1,
      backgroundColor: Colors.amber,
      textColor: Colors.black,
      fontSize: 16.0,
    );
  }

  var message;

  void syncDB() async {
    // SQLite veritabanı bağlantısını oluştur
    DatabaseHelper dbHelper = DatabaseHelper();
    await dbHelper.init(); // init metodunu burada çağır

  }
}
