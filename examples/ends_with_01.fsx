{
  "version": 1,
  "kind": "exercise",
  "title": "Ends with 01",
  "prompt": "Build a DFA over {0, 1} that accepts exactly those words whose last two symbols are 0 then 1. A word shorter than two symbols has no last two, so none of them is accepted -- the empty word included.",
  "alphabet": ["0", "1"],
  "reference": {
    "regex": "(0|1)*01"
  },
  "examples": {
    "accept": ["01", "001", "101", "1101"],
    "reject": ["", "0", "1", "10", "011"]
  }
}
