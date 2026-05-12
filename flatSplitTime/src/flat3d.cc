// updated to make a flat3D output file

/*
  0. Enter the *.ini file, used to decide how to divide the files
  1. Copy the TParameters from in the TFile to the output TFile
  2. Special parmeters that may change: NXY_COARSEN and NETA_COARSEN
  3. Read in the TTree from the input file, coarsen it (if needed) and write
     the segments to the output files
*/


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

#include <fstream>
#include <sstream>
#include <unordered_map>

#include <numeric>


using namespace std;


int main(int argc, char** argv)
{
    const float EPSILON = 1.e-10; // value inserted for negative energy densities
    
    string par_file = "flat3d.ini";
    if (argc>1) { par_file = argv[1]; }

    // default parameters from *.ini file
    std::unordered_map<std::string, int> ini_pars {
        {"NT", 11}, 
        {"MIN_NT", 10}, 
        {"TOFFSET0", 0}, 
        {"TOFFSET1", 0}, 
        {"EVENT_LAST", -1}, 
        {"EVENT_FIRST", 0}, 
        {"IS_TAUEP75", 1},  // boolean
    };
    std::unordered_map<std::string, int> map_int; // maps of input parameters
    std::unordered_map<std::string, float> map_float; // maps of input parameters

    string IFILE_NAME = "/home/davidstewart/xscape-docker/X-SCAPE/build/dAu.root";
    string OFILE_NAME = par_file.substr(0, par_file.find_last_of("."));
    OFILE_NAME += ".root";

    // read *.ini file

    // Read parameters from *.ini file
    std::ifstream par_in(par_file);
    if (par_in.is_open()) {
        std::string line;
        while (std::getline(par_in, line)) {
            // Skip comments
            if (line.find("//") == 0) continue;
            std::istringstream iss(line);
            std::string key, val;
            if (iss >> key >> val) {
                if (key == "IFILE_NAME") {
                    IFILE_NAME = val;
                } else {
                    if (ini_pars.find(key) == ini_pars.end()) {
                        std::cerr << "Warning: Unknown parameter '" << key << "' in ini file." << std::endl;
                    }
                    assert (ini_pars.find(key)!=ini_pars.end());
                    ini_pars[key] = std::stoi(val);
                }
            }
        }
    }

    // ---------------------------------------
    // Get the input 
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

    std::vector<float>* in_dat = nullptr;
    intree->SetBranchAddress("user_res", &in_dat);

    // make the output file
    TFile* fout = TFile::Open(OFILE_NAME.c_str(), "RECREATE");

    // write the ini parameters to the output file as parameters
    for (const auto& key : {"NT", "MIN_NT",  "TOFFSET0", "TOFFSET1", "EVENT_LAST", "EVENT_FIRST", "MINI_EVENTS",
        "IS_TAUEP75"}
    ) {
        TParameter<int> p(key, ini_pars[key]);
        p.Write();
    }

    // write the input file to the output file
    TNamed* p_ifile = new TNamed("IFILE_NAME", IFILE_NAME.c_str());
    p_ifile->Write();
    /* TParameter<string> p_ifile("IFILE_NAME", IFILE_NAME); */
    /* p_ifile.Write(); */
    // read the input parameters, ints and floats, and write to output file
    // int NX, NY, NETA, NFEAT;
    for ( 
            auto& [isint, key] : { 
            std::make_pair(true, "nFeatures"),
            std::make_pair(true, "nx"),
            std::make_pair(false, "dx"),
            std::make_pair(false, "x_min"),
            std::make_pair(true, "ny"),
            std::make_pair(false, "dy"),
            std::make_pair(false, "y_min"),
            std::make_pair(true, "neta"),
            std::make_pair(false, "eta_min"),
            std::make_pair(false, "deta"),
            std::make_pair(false, "tau_min"),
            std::make_pair(true, "ntau"),
            std::make_pair(false, "dtau"),
            }) 
    {
        if (isint) {
            TParameter<int>* p_int;
            infile->GetObject(key, p_int);
            int v_int = p_int->GetVal();
            fout->cd();

            map_int[key] = v_int;

            TParameter<int> p_out(key, v_int);
            p_out.Write();

            /* if (key == "nx") { */
                /* TParameter<int> p_y("ny", v_int); */
                /* p_y.Write(); */
            /* } */
        } else {
            TParameter<float>* p_float;
            infile->GetObject(key, p_float);
            float v_float = p_float->GetVal();

            map_float[key] = v_float;
            std::cout << " read float param " << key << " = " << v_float << std::endl;

            fout->cd();
            TParameter<float> p_out(key, v_float);
            p_out.Write();

            if (key == "x_min") {
                TParameter<float> y_min("y_min", v_float);
                y_min.Write();
            }
            if (key == "dx") {
                TParameter<float> dy("dy", v_float);
                dy.Write();
            }
        }
    }
    int in_ntau {0};
    intree->SetBranchAddress("ntau_freezeout", &in_ntau);

    float in_ftau {0.};
    intree->SetBranchAddress("tau_freezeout", &in_ftau);

    fout->cd();
    TTree* outtree = new TTree("t", "Flat output tree");
    int out_ntau { 0 };
    outtree->Branch("ntau_freezeout", &out_ntau, "ntau_freezeout/I");
    outtree->Branch("tau_freezeout", &in_ftau, "tau_freezeout/F");

    int ntau_start {0};
    outtree->Branch("ntau_start", &ntau_start, "ntau_start/I");

    // const int NXY_COARSEN = ini_pars["NXY_COARSEN"];
    // const int NETA_COARSEN = ini_pars["NETA_COARSEN"];
    // const int NT           = ini_pars["NT"];

    // const bool X_COARSEN   = (NXY_COARSEN > 1);
    // const bool Y_COARSEN   = (NXY_COARSEN > 1);
    // const bool ETA_COARSEN = (NETA_COARSEN > 1);
    // const bool ANY_COARSEN = (X_COARSEN || Y_COARSEN || ETA_COARSEN);

    const bool IS_TAUEP75 = ini_pars["IS_TAUEP75"];

    // don't implement coarsen for now -- leave that to numpy in the ipynb...
    /* assert (!ANY_COARSEN); */

    const int TOFFSET0 = ini_pars["TOFFSET0"];
    const int TOFFSET1 = ini_pars["TOFFSET1"];

    const bool rand_offset = (TOFFSET0 != TOFFSET1);
    const int MIN_NT = ini_pars["MIN_NT"];

    // get some input parmeters
    const int NX = map_int["nx"];
    const int NY = map_int["ny"];
    const int NETA = map_int["neta"];
    const int NFEAT = map_int["nFeatures"];
    const int NT = ini_pars["NT"];

    std::cout << " || " << (map_float.find("tau0") == map_float.end()) << std::endl;
    const float TAU0 = map_float["tau_min"];
    const float DTAU = map_float["dtau"];

    for (auto& [key, val] : map_int) {
        std::cout << " int param: " << key << " = " << val << std::endl;
    }
    for (auto& [key, val] : map_float) {
        std::cout << " float param: " << key << " = " << val << std::endl;
    }


    // in the input vector:
    //  itau     : spaced with NX * NY * NETA * NFEAT
    //  ix       : spaced with      NY * NETA * NFEAT
    //  iy       : spaced with           NETA * NFEAT
    //  ieta     : spaced with                  NFEAT
    //  ifeature : spaced with                        1

    // // make the output tree branch
    const int in_evsize = NX*NY*NETA*NFEAT;

    const int NTOTAL = NX * NY * NETA * NFEAT * ini_pars["NT"];// / (NXY_COARSEN * NXY_COARSEN * NETA_COARSEN);
    float *m_flat = new float[NTOTAL];
    outtree->Branch("flat_data", m_flat, Form("flat_data[%d]/F", NTOTAL));
    const size_t m_size = sizeof(float) * NTOTAL;

    const int EVENT_FIRST = ini_pars["EVENT_FIRST"];
    int EVENT_LAST = ini_pars["EVENT_LAST"];
    int nentries = intree->GetEntries();

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

    TRandom3 rgen(0);
    for (Long64_t i_event = EVENT_FIRST; i_event < EVENT_LAST; ++i_event) {
        intree->GetEntry(i_event);
        if (i_event % 100 == 0) {
            cout << " processing event " << i_event << endl;
        }


        out_ntau = in_dat->size() / in_evsize;

        // check for freezeout frames at the end of the data:
        /* int index_last = out_ntau; */
        while (out_ntau > 0) {
            auto sum_val = std::accumulate(
                in_dat->begin() + (out_ntau - 1) * in_evsize,
                in_dat->begin() + out_ntau * in_evsize,
                0.0f);
            if (sum_val == 0.0) {
                --out_ntau;
            } else {
                break;
            }
        }
        if (out_ntau == 0) {
            std::cerr << " Error: all frames are zero for event " << i_event << std::endl;
            continue;
        }

        if (out_ntau != in_ntau) { 
            /* std::cout << " Warning: ntau_freezeout(" << in_ntau <<") tree != ntau_freezeout in data(" << out_ntau <<"); using later value " << std::endl; */
            std::cout << " || " << (in_ntau - out_ntau) << " frames of zeros removed from end " << std::endl;
        }
        // decide how to cut the event up, each starting at ntau_start
        // total number of time steps: ntau_freezeout
        // std::cout << " in tau " << in_ntau << std::endl;
        // int exsize = NX * NY * NETA * NFEAT * in_ntau;
        // std::cout << " in vector size: " << in_dat->size() << " compared to expected size: " << exsize << " or " << exsize * 1. * (in_ntau )/ (in_ntau+1.) << "  TIME: " << (in_dat->size() / in_evsize ) << std::endl;

        int tau0 = rand_offset ? rgen.Integer(TOFFSET1 - TOFFSET0 + 1) + TOFFSET0 : TOFFSET0;

        for (ntau_start = tau0; ntau_start < out_ntau-MIN_NT; ntau_start += NT) {
            // reset m_flat
            std::memset(m_flat, 0, m_size);
            int itau_end = std::min(ntau_start + NT, out_ntau);
            // std::cout << " TAU range [" << ntau_start <<":"<< itau_end <<") of " << out_ntau << ",  MIN_NT " << MIN_NT << std::endl;

            std::copy(in_dat->data() + ntau_start * in_evsize,
                        in_dat->data() + itau_end * in_evsize,
                        m_flat);
            
            // remove negative energy densities and scale if 
            for (int itau = 0; itau < NT; ++itau) {
                float tau = TAU0 + DTAU * (itau+ntau_start);
                bool print=false;
                const int itau_offset = itau*NX*NY*NETA*NFEAT;
                for (int ix = 0; ix < NX; ++ix) {
                    const int ix_offset = itau_offset + ix*NY*NETA*NFEAT;
                    for (int iy = 0; iy < NY; ++iy) {
                        const int iy_offset = ix_offset + iy*NETA*NFEAT;
                        for (int ieta = 0; ieta < NETA; ++ieta) {
                            const int ieta_offset = iy_offset + ieta*NFEAT;
                            const float eps_old = m_flat[ieta_offset + 0];
                            if (eps_old < EPSILON) {
                                m_flat[ieta_offset] = EPSILON;
                            } else if (IS_TAUEP75) {
                                m_flat[ieta_offset] = tau * pow(eps_old, 0.75);
                            }
                        }
                    }
                }
            }
            outtree->Fill();
        }
    }

    outtree->Write();
    infile->Close();
    fout->Save();
    fout->Close();
    delete[] m_flat;
    return 0;
}
