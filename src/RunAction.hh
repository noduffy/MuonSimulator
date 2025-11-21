#pragma once
#include "G4UserRunAction.hh"
#include <fstream>

class RunAction : public G4UserRunAction {
public:
  RunAction();
  ~RunAction() override;
  void BeginOfRunAction(const G4Run*) override;
  void EndOfRunAction(const G4Run*) override;
  static std::ofstream& Out(); // CSV 共有ストリーム
private:
  static std::ofstream s_out;
};
