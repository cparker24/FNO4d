// This file does the following:
//    input: root file with a TTree with a vector<vector<vector<vector<float>>>> of data
//       optional: max time steps, default 60
//       optional: max events per file, default 1000
//     ouput: file named [input_file]_flat_[nevents]_[nFeatures]_[nXY]_[nTime].root 
//            has a TTree with a flat array of data per event
//            will make multiple files if necessary

// corsen the event to lower resolution in x and y by a factor of 4 and increate time out toward end

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

    // 0: name of input
    // 1: number of events (-1 for all)
    // 2: output file name
    // 3: last time step

    string name_stem = "jetscape_main";
    if (argc>1) { name_stem = argv[1]; }

    int last_event = -1;
    if (argc>2) { last_event = atoi(argv[2]); }

    TString ofname="";
    if (argc>3) { ofname = argv[3]; }

    int nT = 120;
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


    // here is the corsening --> go down by a factor of 4
    const int nXY = (*in_dat)[0].size();
    const int nFeatures = (*in_dat)[0][0][0].size();
    const int nTOTAL = nFeatures * nXY * nXY * nT / 4;

    // make the output tree:
    if (ofname == "") {
        ofname = Form("%s_flat_xy%d_t%d.root", name_stem.c_str(), nXY, nT);
    }
    TFile* outfile = TFile::Open(ofname, "RECREATE");

    TTree* outtree = new TTree("t", "Flat output tree");
    TTree* freezetree = new TTree("freezetree", "Freeze out steps");

    float nsteps = 0;
    freezetree->Branch("nsteps", &nsteps, "nsteps/F");

    // Prepare output container
    float *m_flat = new float[nTOTAL];
    outtree->Branch("user_res", m_flat, Form("user_res[%d]/F", nTOTAL));

    // Keep 

    // outfile->cd();
    // outtree->Write();
    // outfile->Close();

    // make the output data file, flat into the new ttree:
             // from [nX, nY, nT nFeatures] tp [nFeatures, nX, nY, nT]
    // cout << "in 4: " << (*in_dat)[0][0][0][0].size() << endl;
    // for (const auto& vec_)

    const int o_F  =  nXY * nXY * nT / 4;
    const int o_2F  =  2 * nXY * nXY * nT / 4;
    const int o_X  =  nXY * nT;
    const int o_Y  =  nT;

    const int o_iF = nXY * nXY * nT / 4;
    const int o_iX = nXY * nT / 2;
    const int o_iY = nT;

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

        int nX = in_dat->size();
        int nY = (*in_dat)[0].size();
        int nTau = (*in_dat)[0][0].size();
        int nF = (*in_dat)[0][0][0].size();
        nsteps = nTau;
        freezetree->Fill();

        for (int ix = 0; ix < nX; ix+=2) {
            for (int iy=0; iy<nY; iy+= 2) {
                for (int iT=0; iT<nTau; ++iT) {
                    if (iT == nT) break;
                    for (int iF = 0; iF<3; ++iF) {
                        float _sum = (*in_dat)[ix][iy][iT][iF] +
                                     (*in_dat)[ix+1][iy][iT][iF] +
                                     (*in_dat)[ix][iy+1][iT][iF] +
                                     (*in_dat)[ix+1][iy+1][iT][iF];
                        m_flat[iF*o_iF + ix/2 * o_iX + iy/2*o_iY + iT] = _sum/4.;
                    }
                    // float sum_vx  = (*in_dat)[ix][iy][iT][1] + (*in_dat)[ix+1][iy][iT][1] + (*in_dat)[ix][iy+1][iT][1] + (*in_dat)[ix+1][iy+1][iT][1];
                    // float sum_vy  = (*in_dat)[ix][iy][iT][2] + (*in_dat)[ix+1][iy][iT][2] + (*in_dat)[ix][iy+1][iT][2] + (*in_dat)[ix+1][iy+1][iT][2];
                    // m_flat[ix/2]
                }
                for (int iT=nTau; iT<nT; ++iT) {
                    for (int iF = 0; iF<3; ++iF) {
                        m_flat[iF*o_iF + ix/2 * o_iX + iy/2*o_iY + iT] = 0;
                    }
                }
            }
        }
        outtree->Fill(); // fill the output tree with the flat data for this event
        if (last_event >= 0 && i_event >= last_event) {
            cout << "Reached last event: " << last_event << ", stopping." << endl;
            break; // stop processing if we reached the last event
        }
    } // end loop over events
    outfile->cd();
    freezetree->Write();
    outtree->Write();
    outfile->Close();

    infile->Close();
}
