#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction() {
  fGun = new G4ParticleGun(1);
  auto muMinus = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  fGun->SetParticleDefinition(muMinus);
  fGun->SetParticleEnergy(3*GeV);
  fGun->SetParticlePosition({0,0,4*cm});
  fGun->SetParticleMomentumDirection({0,0,-1}); // 下向きに1発
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
  delete fGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  fGun->GeneratePrimaryVertex(event);
}
