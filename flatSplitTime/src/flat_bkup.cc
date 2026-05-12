// This file does the following:
//    input: root file with a TTree with a vector<vector<vector<vector<float>>>> of data
//       optional: max time steps, default 60
//       optional: max events per file, default 1000
//     ouput: file named [input_file]_flat_[nevents]_[nFeatures]_[nXY]_[nTime].root 
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

using namespace std;

int main(int argc, char** argv)
{
    clock_t t; t = clock();
    time_t start, end; time(&start);

    string name_stem = "jetscape_main";
    if (argc>1) { name_stem = argv[1]; }

    int last_event = -1;
    if (argc>2) { last_event = atoi(argv[2]); }

    TString ofname="";
    if (argc>3) { ofname = argv[3]; }


    int nT = 60;
    if (argc>4) { nT = atoi(argv[4]); }

    bool freeze_only = (nT==0);

    /* int maxEvents = 1000; */
    /* if (argc>3) { maxEvents = atoi(argv[3]); } */


    // ---------------------------------------
    // Get the input files
    // ---------------------------------------
    // open the input file and get the ttree
    TFile* infile = TFile::Open((name_stem + ".root").c_str(), "READ");
    if (!infile || infile->IsZombie()) {
        std::cerr << "Error opening file: " << name_stem << ".root" << std::endl;
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


    const int nXY = (*in_dat)[0].size();
    const int nFeatures = (*in_dat)[0][0][0].size();
    const int nTOTAL = nFeatures * nXY * nXY * nT;

    // make the output tree:
    if (ofname == "") {
        ofname = Form("%s_flat_xy%d_t%d.root", name_stem.c_str(), nXY, nT);
    }
    TFile* fout = TFile::Open(ofname, "RECREATE");

    TTree* outtree = new TTree("t", "Flat output tree");
    TTree* freezetree = new TTree("freezetree", "Freeze out steps");

    float nsteps = 0;
    freezetree->Branch("nsteps", &nsteps, "nsteps/F");

    // Prepare output container
    float *m_flat = new float[nTOTAL];
    outtree->Branch("user_res", m_flat, Form("user_res[%d]/F", nTOTAL));

    // Keep 

    // outtree->Write();
    // fout->Close();

    // make the output data file, flat into the new ttree:
    // from [nX, nY, nT nFeatures] tp [nFeatures, nX, nY, nT]
    // cout << "in 4: " << (*in_dat)[0][0][0][0].size() << endl;
    // for (const auto& vec_)

    // in the flat space, there is:
    //   iFeatures : spaced with nXY*nXY*nT
    //   iX        : spaced with     nXY*nT
    //   iY        : spaced with         nT

    const int o_X  =  nXY * nT;
    const int o_Y  =  nT;
    const int o_F  =  nXY * nXY * nT;
    const int o_2F  =  2 * nXY * nXY * nT;

    for (Long64_t i_event = 0; i_event < nentries; ++i_event) {
        intree->GetEntry(i_event);
        if (i_event % 100 == 0) {
            cout << " processing event " << i_event << endl;
        }

        if (freeze_only) {
            // only fill the freeze tree
            nsteps = (*in_dat)[0][0].size();
            freezetree->Fill();
            if (last_event >= 0 && i_event >= last_event) {
                cout << "Reached last event: " << last_event << ", stopping." << endl;
                break; // stop processing if we reached the last event
            }
            continue; // skip to the next event
        }

        bool first_vec_set = true;

        int i_X = 0; // the offset due to x
        for (const auto& vec_ytf : *in_dat) {
            int i_Y = i_X; // the offset due to y
            for (const auto& vec_tf : vec_ytf) {
                if (first_vec_set) {
                    nsteps = vec_tf.size(); // get the number of steps from the first event
                    freezetree->Fill();
                    first_vec_set = false;
                }
                int i_T = i_Y;
                int cnt_T = 0;
                for (const auto& vec_f : vec_tf) {
                    m_flat[i_T] = vec_f[0];
                    m_flat[i_T + o_F] = vec_f[1];
                    m_flat[i_T + o_2F] = vec_f[2];
                    i_T += 1; // increment time offset
                    cnt_T += 1;
                    if (cnt_T >= nT) { break; }
                } // end loop over time+featuers
                for (int i_zeros = cnt_T; i_zeros < nT; ++i_zeros) {
                    // fill in zero values for the rest of the time steps
                    m_flat[i_T] = 0.;
                    m_flat[i_T + o_F] = 0.;
                    m_flat[i_T + o_2F] = 0.;

                    i_T += 1; // increment time offset
                } // end zero's past end of time steps
                i_Y += o_Y;
            } // end loop over y
            i_X += o_X;
        } // end loop over x
        float vmax = 0;
        for (int i=0; i<nTOTAL; ++i) { if (m_flat[i] > vmax) vmax = m_flat[i]; }
        outtree->Fill(); // fill the output tree with the flat data for this event
        if (last_event >= 0 && i_event >= last_event) {
            cout << "Reached last event: " << last_event << ", stopping." << endl;
            break; // stop processing if we reached the last event
        }
    } // end loop over events
    fout->cd();
    freezetree->Write();
    outtree->Write();
    fout->Close();

    //     int nT_in = (*in_dat)[0][0].size();
    //     for (int iX = 0; iX < nXY; ++iX) {
    //         for (int iY = 0; iY < nXY; ++iY) {
    //             for (int iT = 0; iT < nT; ++iT) {
    //                 if (iT >= nT_in) {
    //                     // fill in zero values
    //                 } else {
    //                     // copy over the actual values
    //                 }
    //             }
    //         }
    //     }
    //     break; // break for testing
    // }

    infile->Close();
}
