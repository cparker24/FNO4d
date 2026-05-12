// This file does the following:
//    input: root file with a TTree with a vector<vector<vector<vector<float>>>> of data
//       optional: max time steps, default 60
//       optional: max events per file, default 1000
//     ouput: file named [input_file]_flat_[nevents]_[nfeatures]_[nXY]_[nTime].root 
//            has a TTree with a flat array of data per event
//            will make multiple files if necessary


#include <iostream>
#include <time.h>
#include <string>

#include <chrono>
#include <thread>

#include "TFile.h"
#include "TString.h"
#include "TTree.h"

#include "TParameter.h"
#include "TRandom3.h"

using namespace std;

namespace {
    inline void apply_tauep75(float& e_in, int tau_index) {
    e_in = (0.5+tau_index/10.)*pow(e_in,0.75);
    };
}

int main(int argc, char** argv)
{

    
    string par_file = "flat.ini";
    if (argc>1) { par_file = argv[1]; }

    // parameters to be read in
    int NT = 60;
    int MIN_NT = 10;
    int N_COARSEN = 1;
    int TOFFSET0 = 0;
    int TOFFSET1 = 0;
    int EVENT_LAST  = -1;
    int EVENT_FIRST = 0;
    bool MINI_EVENTS = false;
    bool IS_TAUEP75 = true;
    string IFILE_NAME = "jetscape_main.root";
    string OFILE_NAME = par_file.substr(0, par_file.find_last_of("."));
    OFILE_NAME += ".root";
    // string OFILE_NAME = in_stem + ".root";

    //
    // string par_file = in_stem + ".ini";
    #include <fstream>
    #include <sstream>
    #include <map>

    // Read parameters from par_file
    std::ifstream par_in(par_file);
    if (par_in.is_open()) {
        std::string line;
        while (std::getline(par_in, line)) {
            // Skip comments
            if (line.find("//") == 0) continue;
            std::istringstream iss(line);
            std::string key, val;
            if (iss >> key >> val) {
                if (key == "NT") {
                    NT = std::stoi(val);
                } else if (key == "MIN_NT") {
                    MIN_NT = std::stoi(val);
                // } else if (key == "N_COARSEN") {
                    // N_COARSEN = std::stoi(val);
                } else if (key == "TOFFSET0") {
                    TOFFSET0 = std::stoi(val);
                } else if (key == "TOFFSET1") {
                    TOFFSET1 = std::stoi(val);
                } else if (key == "MINI_EVENTS") {
                    MINI_EVENTS = (val == "true" || val == "1");
                } else if (key == "IFILE_NAME") {
                    IFILE_NAME = val;
                } else if (key == "EVENT_LAST") {
                    EVENT_LAST = std::stoi(val);
                } else if (key == "EVENT_FIRST") {
                    EVENT_FIRST = std::stoi(val);
                } else if (key == "IS_TAUEP75") {
                    IS_TAUEP75 = std::stoi(val);
                }
            }
        }
    }

    // logic:
    // read in the events
    // write out the events to the output ROOT file
    // same the parameters for nfeatures, NX, NT, so that the output events
    // can be re-shaped into tensors of sizes [nfeatures, NX, NX, NT]

    // ---------------------------------------
    // Get the input files
    // ---------------------------------------
    // open the input file and get the ttree
    TFile* infile = TFile::Open(IFILE_NAME.c_str(), "READ");
    if (!infile || infile->IsZombie()) {
        std::cerr << "Error opening file: " << par_file << ".root" << std::endl;
        return 1;
    }

    TTree* intree = nullptr;
    infile->GetObject("t", intree);
    if (!intree) {
        std::cerr << "Error: TTree 't' not found in file." << std::endl;
        infile->Close();
        return 1;
    }

    std::vector<std::vector<std::vector<std::vector<float>>>>* in_dat = nullptr;
    intree->SetBranchAddress("user_res", &in_dat);

    Long64_t nentries = intree->GetEntries();
    cout << "Number of entries in the tree: " << nentries << endl;
    intree->GetEntry(0);


    TFile* fout = TFile::Open(OFILE_NAME.c_str(), "RECREATE");
    TTree* outtree = new TTree("t", "Flat output tree");
    int itau_FO = 0;
    outtree->Branch("itau_freezeout", &itau_FO, "itau_freezeout/I");
    int itau = 0;
    outtree->Branch("itau", &itau, "itau/I"); // first time step per event
    int nfrozen = 0;
    outtree->Branch("ntau_frozen", &nfrozen, "ntau_frozen/I"); // first time step per event



    // make the output data file, flat into the new ttree:
    // from [nX, nY, nT, nfeatures] to [nfeatures, nX, nY, nT]
    // cout << "in 4: " << (*in_dat)[0][0][0][0].size() << endl;
    // for (const auto& vec_)

    // in the flat space, there is:
    //   iFeatures : spaced with nXY*nXY*nT
    //   iX        : spaced with     nXY*nT
    //   iY        : spaced with         nT

    // const int o_X  =  nXY * nT;
    // const int o_Y  =  nT;
    // const int o_F  =  nXY * nXY * nT;
    // const int o_2F  =  2 * nXY * nXY * nT;

    bool first_event;
    // bool coarsen = (N_COARSEN > 1);
    bool toffset = (TOFFSET0 > 0 || TOFFSET1 > 0);

    TRandom3 rgen(0);

    intree->GetEntry(0);
    const int in_nXY = p_NX; //(*in_dat).size();
    const int nfeatures = 3; // (*in_dat)[0][0][0].size();
    const int nXY = in_nXY / N_COARSEN;

    const int mult_F = nXY * nXY * NT;
    const int mult_X =       nXY * NT;
    const int mult_Y =             NT;

    fout->cd();

    TParameter<int> p_NT("NT", NT);
    p_NT.Write();

    // TParameter<int> p_IS_TAUEP75("IS_TAUEP75", IS_TAUEP75);
    // p_IS_TAUEP75.Write();

    // TParameter<int> p_NX("NX", nXY);
    // p_NX.Write();

    // TParameter<int> p_MIN_NT("MIN_NT", MIN_NT);
    // p_MIN_NT.Write();

    TParameter<int> p_NCOARSEN("N_COARSEN", N_COARSEN);
    p_NCOARSEN.Write();

    TParameter<int> p_NFEATURES("N_FEATURES", nfeatures);
    p_NFEATURES.Write();

    TParameter<int> p_TOFFSET0("TOFFSET0", TOFFSET0);
    p_TOFFSET0.Write();

    TParameter<int> p_TOFFSET1("TOFFSET1", TOFFSET1);
    p_TOFFSET1.Write();

    // make the output tree branch
    const int ntotal = nfeatures*nXY*nXY*NT;
    float *m_flat = new float[ntotal];
    outtree->Branch("user_res", m_flat, Form("user_res[%d]/F", ntotal));

    if (EVENT_FIRST > nentries) {
        cout << " First event " << EVENT_FIRST  << " is greater than total events " << nentries << endl;
        cout << " Ending program." << endl;
        return 0;
    }
    if (EVENT_LAST < 1) {
        EVENT_LAST = nentries;
    } else if (EVENT_LAST > nentries) {
        cout << " Last event at " << EVENT_LAST << " is greater than total events " << nentries << endl;
        cout << " -> Setting last event to " << nentries<< endl;
        EVENT_LAST = nentries;
    } 

    for (Long64_t i_event = EVENT_FIRST; i_event < EVENT_LAST; ++i_event) {
        intree->GetEntry(i_event);
        if (i_event % 100 == 0) {
            cout << " processing event " << i_event << endl;
        }
        itau_FO = (*in_dat)[0][0].size();
        if (itau_FO < MIN_NT) continue;

        itau = (toffset) ? rgen.Integer(TOFFSET1-TOFFSET0+1) + TOFFSET0 : 0;
        int nmini = 1;
        if (MINI_EVENTS) {
            int ntau_total = itau_FO-itau;
            nmini = ntau_total / NT;
            if ((ntau_total % NT) > MIN_NT) {
                // will buffer the final event with 0.'s after freezeout to fill up
                nmini += 1;
            }
        }
        for (int imini = 0; imini < nmini; ++imini) {
            nfrozen = max(itau+NT-itau_FO, 0);
            int ntau = NT - nfrozen;
            for (int iX = 0; iX<in_nXY; ++iX) {
            for (int iY = 0; iY<in_nXY; ++iY) {
            for (int iF=0; iF<nfeatures; ++iF) {
                for (int iT=0; iT<ntau; ++iT) {
                    float val = (*in_dat)[iX][iY][iT+itau][iF]; 
                    if (iF==0 && IS_TAUEP75) apply_tauep75(val, iT+itau);
                    m_flat[iF*mult_F+iX*mult_X+iY*mult_Y+iT] = val;
                }
                for (int iT=ntau; iT<NT; ++iT) {
                m_flat[iF*mult_F + iX*mult_X + iY*mult_Y + iT] = 0.;
                }
            }}} 
            outtree->Fill();
            itau += NT;
        }
    } // end loop over events

    fout->cd();
    outtree->Write();
    fout->Close();
    infile->Close();
}
