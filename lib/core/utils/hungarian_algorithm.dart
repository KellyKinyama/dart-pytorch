/// Hungarian (Kuhn–Munkres) algorithm for minimum-cost assignment.
///
/// Given an `n × n` cost matrix, `HungarianAlgorithm(costs).getAssignment()`
/// returns a `List<int>` where `result[agent] = task` is the task
/// assigned to each agent in the optimal (minimum total cost)
/// bipartite matching. If the underlying problem is rectangular
/// (fewer real GT objects than predicted queries), pad with a large
/// sentinel cost so unmatched rows get pushed to the pad columns.
///
/// Pure Dart, host-side. Used by the DETR-style object-detection
/// demos to align predicted query slots to ground-truth boxes.
library;

import 'dart:collection';
import 'dart:math';

class HungarianAlgorithm {
  late List<List<int>> _cost;
  late int _n;

  late List<int> _xy;
  late List<int> _yx;

  late List<int> _lx;
  late List<int> _ly;

  late List<bool> _inTreeX;
  late List<bool> _inTreeY;
  late List<int> _prev;
  late List<int> _slack;
  late List<int> _slackX;

  int _matchCount = 0;

  HungarianAlgorithm(List<List<int>> costMatrix) {
    _n = costMatrix.length;
    _cost = List.generate(_n, (i) => List<int>.from(costMatrix[i]));
    // Convert cost -> profit (max-weight matching).
    for (int i = 0; i < _n; i++) {
      for (int j = 0; j < _n; j++) {
        _cost[i][j] *= -1;
      }
    }
    _xy = List.filled(_n, -1);
    _yx = List.filled(_n, -1);
    _lx = List.filled(_n, 0);
    _ly = List.filled(_n, 0);
    _slack = List.filled(_n, 0);
    _slackX = List.filled(_n, 0);
    _prev = List.filled(_n, 0);
    _inTreeX = List.filled(_n, false);
    _inTreeY = List.filled(_n, false);
  }

  void _labelIt() {
    for (int i = 0; i < _n; i++) {
      for (int j = 0; j < _n; j++) {
        _lx[i] = max(_lx[i], _cost[i][j]);
      }
    }
  }

  void _addTree(int x, int prevX) {
    _inTreeX[x] = true;
    _prev[x] = prevX;
    for (int y = 0; y < _n; y++) {
      final currentSlack = _lx[x] + _ly[y] - _cost[x][y];
      if (currentSlack < _slack[y]) {
        _slack[y] = currentSlack;
        _slackX[y] = x;
      }
    }
  }

  void _updateLabels() {
    int delta = 999999999;
    for (int y = 0; y < _n; y++) {
      if (!_inTreeY[y]) delta = min(delta, _slack[y]);
    }
    for (int x = 0; x < _n; x++) {
      if (_inTreeX[x]) _lx[x] -= delta;
    }
    for (int y = 0; y < _n; y++) {
      if (_inTreeY[y]) _ly[y] += delta;
    }
    for (int y = 0; y < _n; y++) {
      if (!_inTreeY[y]) _slack[y] -= delta;
    }
  }

  void _augment() {
    if (_matchCount == _n) return;

    int x, y = -1, root = -1;
    final q = Queue<int>();

    for (int i = 0; i < _n; i++) {
      if (_xy[i] == -1) {
        q.add(root = i);
        _prev[i] = -2;
        _inTreeX[i] = true;
        break;
      }
    }
    if (root == -1) return;

    for (int i = 0; i < _n; i++) {
      _slack[i] = _lx[root] + _ly[i] - _cost[root][i];
      _slackX[i] = root;
    }

    while (true) {
      while (q.isNotEmpty) {
        x = q.removeFirst();
        for (y = 0; y < _n; y++) {
          if ((_lx[x] + _ly[y] - _cost[x][y] == 0) && (!_inTreeY[y])) {
            if (_yx[y] == -1) {
              break;
            } else {
              _inTreeY[y] = true;
              q.add(_yx[y]);
              _addTree(_yx[y], x);
            }
          }
        }
        if (y < _n) break;
      }
      if (y < _n) break;

      _updateLabels();

      for (y = 0; y < _n; y++) {
        if (!_inTreeY[y] && _slack[y] == 0) {
          if (_yx[y] == -1) {
            x = _slackX[y];
            break;
          } else {
            _inTreeY[y] = true;
            if (!_inTreeX[_yx[y]]) {
              q.add(_yx[y]);
              _addTree(_yx[y], _slackX[y]);
            }
          }
        }
      }
      if (y < _n) break;
    }

    if (y < _n) {
      _matchCount++;
      for (int cx = _slackX[y], cy = y, ty; cx != -2; cx = _prev[cx], cy = ty) {
        ty = _xy[cx];
        _xy[cx] = cy;
        _yx[cy] = cx;
      }
      _inTreeX = List.filled(_n, false);
      _inTreeY = List.filled(_n, false);
      _augment();
    }
  }

  /// Returns the minimum-cost assignment: `result[agent] = task`.
  List<int> getAssignment() {
    _labelIt();
    while (_matchCount < _n) {
      _inTreeX = List.filled(_n, false);
      _inTreeY = List.filled(_n, false);
      _prev = List.filled(_n, 0);
      _slack = List.filled(_n, 999999999);
      _slackX = List.filled(_n, 0);
      final initial = _matchCount;
      _augment();
      if (_matchCount == initial && _matchCount < _n) break;
    }
    return _xy;
  }
}
