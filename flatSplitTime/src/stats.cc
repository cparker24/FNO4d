// Read through the input file, and make a TTree with the number of time steps and the total starting energy (rho times entries) in the IC


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

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        cout << "Usage: " << argv[0] << " [name_stem] [output_file] [last_event]" << endl;
        cout << "  name_stem: Base name for input file (default: jetscape_main)" << endl;
        cout << "  last_event: total number of events to process (default: all events)" << endl;
        cout << "  out file name: Name of the output file (default: <name_stem>_stats.root)" << endl;
        return 0;
    }

    string name_stem = "jetscape_main";
    if (argc>1) { name_stem = argv[1]; }

    int last_event = -1;
    if (argc>2) { last_event = atoi(argv[2]); }

    TString ofname="";
    if (argc>3) { ofname = argv[3]; }



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

    bool first_evnent = true;

    float nX = 0;
    float nY = 0;

    // make the output tree:
    if (ofname == "") {
        TString name_stem_only = name_stem;
        int last_slash = name_stem_only.Last('/');
        if (last_slash != kNPOS) {
            name_stem_only = name_stem_only(last_slash + 1, name_stem_only.Length() - last_slash - 1);
        }
        int last_dot = name_stem_only.Last('.');
        if (last_dot != kNPOS) {
            name_stem_only = name_stem_only(0, last_dot);
        }

        ofname = Form("%s_stats.root", name_stem_only.Data());
        cout << " ofname: " << ofname << endl;
    } else {
        cout << " This is the oname |" << ofname << "|" << endl;
    }

    TFile* outfile = TFile::Open(ofname, "RECREATE");
    TTree* outtree = new TTree("t", "Flat output tree");
    // TTree* freezetree = new TTree("freezetree", "Freeze out steps");

    float nsteps = 0;
    double sum_erho = 0.;
    outtree->Branch("nsteps", &nsteps, "nsteps/F");
    outtree->Branch("sum_erho", &sum_erho, "sum_erho/D");
    bool first_event = true;

    for (Long64_t i_event = 0; i_event < nentries; ++i_event) {
        intree->GetEntry(i_event);
        if (i_event % 100 == 0) {
            cout << " processing event " << i_event << endl;
        }

        if (first_event) {
            nX = (*in_dat)[0].size();
            nY = (*in_dat)[0][0].size();
            first_event = false;
        }


        // only fill the freeze tree
        nsteps = (*in_dat)[0][0].size();
        sum_erho = 0.;

        // sum the energy entries for the first time step
        for (const auto& vec_ytf : *in_dat) {
            for (const auto& vec_tf : vec_ytf) {
                sum_erho += vec_tf[0][0]; // sum the first feature (energy)

                // for test output on the velocities:
                if (vec_tf[0][1] > 0.000001 || vec_tf[0][2] > 0.000001) {

                    cout  << " WARMING: " << endl
                        << "Event " << i_event << ": "
                            << "Energy: " << vec_tf[0][0] << ", "
                            << "Velocity: (" << vec_tf[0][1] << ", " << vec_tf[0][2] << ")"
                            << endl;
                }
            }
        }
        sum_erho /= (nX*nY);
        outtree->Fill();

        if (last_event >= 0 && i_event >= last_event) {
            cout << "Reached last event " << last_event << ", stopping." << endl;
            break;
        }
    } // end loop over events
    outfile->cd();
    outtree->Write();
    outfile->Close();

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
