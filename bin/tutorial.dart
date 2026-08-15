// Toy dataset of sentences
import 'dart:math';

final documents = [
  "Machine learning is powerful",
  "Artificial intelligence advances rapidly",
  "Deep learning transforms technology",
  "Data science drives innovation",
  "Neural networks power AI",
];
// Simplified 2D word embeddings (in practice, use pre-trained embeddings)
final word_embeddings = {
  "machine": [0.8, 0.2],
  "learning": [0.7, 0.3],
  "powerful": [0.6, 0.4],
  "artificial": [0.9, 0.1],
  "intelligence": [0.85, 0.15],
  "advances": [0.5, 0.5],
  "rapidly": [0.4, 0.6],
  "deep": [0.75, 0.25],
  "transforms": [0.65, 0.35],
  "technology": [0.7, 0.4],
  "data": [0.3, 0.7],
  "science": [0.35, 0.65],
  "drives": [0.4, 0.6],
  "innovation": [0.45, 0.55],
  "neural": [0.8, 0.2],
  "networks": [0.78, 0.22],
  "power": [0.6, 0.4],
  "ai": [0.9, 0.1],
};

List<String> tokenize(String text) {
  /// Convert text to lowercase and split into words.
  return RegExp(
    r'\w+',
  ).allMatches(text.toLowerCase()).map((m) => m.group(0)!).toList();
}

List<double> sentenceToVector(
  String sentence,
  Map<String, List<double>> embeddings,
) {
  /// Convert a sentence to a vector by averaging word embeddings.
  List<String> words = tokenize(sentence);
  List<List<double>> vectors = words
      .map((word) => embeddings[word] ?? [0.0, 0.0])
      .toList();
  vectors = vectors
      .where((v) => v.fold(0.0, (sum, val) => sum + val) != 0)
      .toList();
  if (vectors.isEmpty) {
    return [0.0, 0.0]; // Return zero vector if no valid words
  }
  int dimensions = vectors[0].length;
  List<double> result = List.filled(dimensions, 0.0);
  for (int i = 0; i < dimensions; i++) {
    for (List<double> v in vectors) {
      result[i] += v[i];
    }
    result[i] /= vectors.length;
  }
  return result;
}

// Convert all documents to vectors
List<List<double>> docVectors = documents
    .map((doc) => sentenceToVector(doc, word_embeddings))
    .toList();

double cosineSimilarity(List<double> vec1, List<double> vec2) {
  /// Compute cosine similarity between two vectors.
  double dotProduct = 0.0;
  for (int i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
  }

  double norm1 = 0.0;
  double norm2 = 0.0;
  for (int i = 0; i < vec1.length; i++) {
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  norm1 = norm1 > 0 ? sqrt(norm1) : 0.0;
  norm2 = norm2 > 0 ? sqrt(norm2) : 0.0;

  if (norm1 == 0.0 || norm2 == 0.0) {
    return 0.0;
  }
  return dotProduct / (norm1 * norm2);
}

List<(String, double)> vectorSearch(
  String query,
  List<String> documents,
  Map<String, List<double>> embeddings, [
  int topK = 3,
]) {
  /// Perform vector search and return top-k similar documents.
  List<double> queryVector = sentenceToVector(query, embeddings);
  List<double> similarities = docVectors
      .map((docVec) => cosineSimilarity(queryVector, docVec))
      .toList();

  /// Get indices of top-k similarities
  List<int> rankedIndices = List.generate(similarities.length, (i) => i);
  rankedIndices.sort((a, b) => similarities[b].compareTo(similarities[a]));
  rankedIndices = rankedIndices.take(topK).toList();

  List<(String, double)> results = [
    for (int i in rankedIndices)
      if (similarities[i] > 0) (documents[i], similarities[i]),
  ];
  return results;
}

void main() {
  /// Example query
  String query = "Machine learning technology";
  List<(String, double)> results = vectorSearch(
    query,
    documents,
    word_embeddings,
  );
  print("Query: $query");
  print("Top results:");
  for (var (doc, score) in results) {
    print("Score: ${score.toStringAsFixed(3)}, Document: $doc");
  }
}
