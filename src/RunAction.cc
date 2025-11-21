#include "RunAction.hh"
#include "G4Run.hh"
#include <iomanip>

std::ofstream RunAction::s_out;

RunAction::RunAction() = default;
RunAction::~RunAction() = default;

void RunAction::BeginOfRunAction(const G4Run*) {
  if (!s_out.is_open()) {
    s_out.open("hits.csv", std::ios::out | std::ios::trunc);
    s_out << "event,track,det,zname,x,y,z,dx,dy,dz\n"; // ヘッダ
    s_out.flush();
  }
}

void RunAction::EndOfRunAction(const G4Run*) {
  if (s_out.is_open()) s_out.flush();
}

std::ofstream& RunAction::Out() { return s_out; }
