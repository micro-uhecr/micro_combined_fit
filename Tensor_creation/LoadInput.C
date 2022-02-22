#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

//ROOT based
#include "TSystem.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TH1D.h"

using namespace std;

//Utility for sorting//////////////////////////////
template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);
	stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
	return idx;
}

//Load SimProp text files for a fixed mass A///////
//each file corresponds to a given redshift range//
vector<string> LoadSimPropFiles(int A, string input_filename_prefix){
	vector<string> vfiles; 
	
	stringstream cmd;
	cmd<<"ls "<<input_filename_prefix<<A<<"_*.txt | wc -l";
	unsigned int nfiles = atoi(gSystem->GetFromPipe(cmd.str().c_str()));
	for(unsigned int i=0; i<nfiles; i++){
		stringstream tmpcmd;
		tmpcmd<<"ls "<<input_filename_prefix<<A<<"_*.txt | sed -n '"<<i+1<<"p'";
		string tmpfile(gSystem->GetFromPipe(tmpcmd.str().c_str()));
		if(tmpfile.size()>0) vfiles.push_back(tmpfile);
	}
	
	return vfiles;
}

//Export the data in intermediate output folder////
//each file corresponds to a given redshift range//
//note: precision params -> numerical precision////
//with which these numbers are stored//////////////
void ExportDistrib(int A, string input_filename_prefix, string output_filename_prefix, double zmax, double logRmin, double logRmax, int precisionR = 1000, int precisionzw = 10000, bool verbose=false){

	//Load data
	vector<string> vfiles = LoadSimPropFiles(A, input_filename_prefix);
	
	//Write output
	stringstream sfileout;
	sfileout<<output_filename_prefix<<A<<"_zmax_"<<zmax<<".dat";
	ofstream myfile_out;
	myfile_out.open(sfileout.str().c_str());
	myfile_out<<"id z logRi wi Z A logE w\n";

	vector< double > vlogRi, vzi, vwi;		
	for(unsigned int i=0; i<vfiles.size(); i++){
		cout<<"Reading file: "<<vfiles[i]<<endl;
		ifstream file(vfiles[i].c_str());
		
		int line_number = 0;
		while(!file.eof()){
			line_number++;
			
			//input parameters
			int Ai, Zi, id;
			double Ei, zi;
			//number of daughter nuclei
			unsigned int nNuc;
			file >> id >> Zi >> Ai >> Ei >> zi >> nNuc;//How the data is stored

			//weight applied to get a uniform distribution in z
			double one_p_z3 = TMath::Power(1+zi,3);
			double dzdt = (1+zi)*TMath::Sqrt(0.7+0.3*one_p_z3);//standard LambdaCDM cosmology used to generate the sim prop files
			
			//Loop on daugther nuclei
			vector< double > vEd, vwd;
			vector< int > vAd, vZd;
			double wgen;
			for(unsigned int j=0; j<nNuc; j++){
				double Ed;
				int Ad, Zd;
				file >> Zd >> Ad >> Ed >> wgen;//note wgen is the same for all daughter nuclei
				if((Ad<=A) and (Ad>=0)){//check to avoid crazy entries
					vEd.push_back(Ed);
					vwd.push_back(wgen*dzdt);
					vAd.push_back(Ad);
					vZd.push_back(Zd);
				}
				else if(verbose) cout<<"Wrong daughter nuclei on line "<<line_number<<" in file "<<vfiles[i]<<" !!!!!!!!!!!!!"<<endl;
			}
			
			double ri = TMath::Log10(Ei/Zi);//stored as initial rigidity
			if((zi<=zmax) && (ri>=logRmin) && (ri<=logRmax) ){//storage up to zmax, between logRmin and logRmax					
				//relevant parameters on Earth
				vector< int > vA, vZ;
				vector< double > vE, vw;
				double weight = 0;
				//loop on entries sorted by A of daughter particles
				for (auto j: sort_indexes(vAd)) {
					double ed = TMath::Log10(vEd[j]);
					double e = std::round(ed*precisionR)/precisionR;
					weight = vwd[j];
					//cumulate identical daughter nuclei
					bool are_the_same = (vA.size()>0) and (TMath::Abs(vA.back()-vAd[j])<1) and (TMath::Abs(vZ.back()-vZd[j])<1) and (TMath::Abs(e-vE.back())<=1.0/precisionR);
					if(not are_the_same){
						vA.push_back(vAd[j]);
						vZ.push_back(vZd[j]);;
						vE.push_back(e);
						vw.push_back(weight);			
					}
					else vw.back()+= weight;
				}
				//write
				for(unsigned int j=0; j<vA.size(); j++){
					myfile_out << i<<"_"<<id << " "				
					 << std::round(zi*precisionzw)/precisionzw<< " "
					 << std::round(ri*precisionR)/precisionR << " "
 					 << std::round(weight)<< " "
					 << vZ[j] << " "
					 << vA[j] << " "
					 << vE[j] << " "
					 << std::round(vw[j]) <<"\n";
				}
				
				vlogRi.push_back(ri);
				vzi.push_back(zi);
				vwi.push_back(weight);
			}
		}
	}
	myfile_out.close();	
	
	std::cout<<"Number of entries: "<<vlogRi.size()<<std::endl;
	
	//Control plot
	stringstream title;
	title<<"A = "<<A;
	int nbins = 100;

	TH1D *hR = new TH1D("hR",title.str().c_str(),nbins, logRmin, logRmax);	
	hR->SetMinimum(0);
	hR->SetStats(0);		
	hR->GetXaxis()->SetTitle("log10(R[eV])");
	hR->GetXaxis()->CenterTitle();
	hR->GetXaxis()->SetTitleSize(0.05);
	hR->GetXaxis()->SetLabelSize(0.05);	
	hR->GetYaxis()->SetLabelSize(0.05);		
	for(unsigned int i=0; i<vlogRi.size(); i++) hR->Fill(vlogRi[i], vwi[i]);
	
	TH1D *hz = new TH1D("hz",title.str().c_str(),nbins, 0, zmax);	
	hz->SetMinimum(0);
	hz->SetStats(0);	
	hz->GetXaxis()->SetTitle("Redshift, z");
	hz->GetXaxis()->CenterTitle();
	hz->GetXaxis()->SetTitleSize(0.05);
	hz->GetXaxis()->SetLabelSize(0.05);	
	hz->GetYaxis()->SetLabelSize(0.05);	
	for(unsigned int i=0; i<vzi.size(); i++) hz->Fill(vzi[i], vwi[i]);

	TCanvas* c1 = new TCanvas("c1","c1",800,800);
	c1->Divide(1,2);
	c1->cd(1);
	hR->Draw("histe");
	c1->cd(2);
	hz->Draw("histe");
	stringstream spdfout;
	spdfout<<output_filename_prefix<<title.str()<<".pdf";
	c1->SaveAs(spdfout.str().c_str());
}


void LoadInput(){
	//Input and output
	string input_filename_prefix = "Input_Simulations/SimProp_";
	string output_filename_prefix = "Intermediate_MergedInput/SimProp_A_";	
	
	//Parameters that need to be consistent between input and output 
	double zmax = 2.5;//maximum redshift up to which the simulation is performed
	double logRmin = 17, logRmax = 21;//rigidity range in log10(eV)
	
	//vector<int> vA = {1, 4, 12, 14, 16, 20, 24, 28, 32, 56};//e.g. for the full list of species
	vector<int> vA = {12};//single species, Carbon here for A = 12
	for(unsigned int i=0; i<vA.size(); i++){
		std::cout<<"---- "<<vA[i]<<" ----"<<std::endl;
		ExportDistrib(vA[i], input_filename_prefix, output_filename_prefix, zmax, logRmin, logRmax);
	}
}

