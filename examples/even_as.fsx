{
  "version": 1,
  "kind": "exercise",
  "title": "An even number of a's",
  "prompt": "Build a DFA over {a, b} that accepts exactly those words containing an even number of a's. Zero is even, so the empty word is accepted, and there is no bound on the number of b's anywhere in the word.",
  "alphabet": ["a", "b"],
  "reference": {
    "regex": "b*(ab*ab*)*"
  },
  "examples": {
    "accept": ["", "b", "aa", "abba"],
    "reject": ["a", "ba", "abb", "aaa"]
  }
}
