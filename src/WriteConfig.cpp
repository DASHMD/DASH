#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>
#include "WriteConfig.h"
#include "includeFixes.h"

#define BUFFERLEN 700

using namespace std;
namespace py = boost::python;

template <typename T>
void writeXMLChunk(ofstream &outFile, vector<T> &vals, string tag, std::function<void (T &, char [BUFFERLEN])> getLine) {

    char buffer[BUFFERLEN];
    outFile << "<" << tag << ">\n";
    for (T &v : vals) {
        getLine(v, buffer);
        outFile << buffer;
    }
    outFile << "</" << tag << ">\n";
}

template <typename T, typename K>
void writeXMLChunkBase64(ofstream &outFile, vector<T> &vals, string tag, std::function<K (T &)> getItem) {
    vector<K> slicedVals;
    slicedVals.reserve(vals.size());
    for (T &v : vals) {
        slicedVals.push_back(getItem(v));
    }
    vector<unsigned char> copied = vector<unsigned char>((unsigned char *) slicedVals.data(), (unsigned char *) slicedVals.data() + (int) (sizeof(K)/sizeof(unsigned char) * slicedVals.size()));
    string base64 = base64_encode(copied.data(), copied.size());  
    outFile << "<" << tag << " base64=\"1\">\n";
    outFile << base64.c_str();
    outFile << "</" << tag << ">\n";
}


void writeAtomParams(ofstream &outFile, AtomParams &params) {
    
    outFile << "<atomParams numTypes=\"" << params.numTypes << "\">\n";
    outFile << "<handle>\n";
    for (string handle : params.handles) {
        outFile << handle << "\n";
    }
    outFile << "</handle>\n";

    outFile << "<mass>\n";
    for (double mass : params.masses) {
        outFile << mass << "\n";
    }
    outFile << "</mass>\n";


    outFile << "</atomParams>\n";

}

void outputGroups(ofstream &outFile, State *state) {
    outFile << "<groupInfo>\n";
    outFile << "<groupHandles>\n";
    for (auto it = state->groupTags.begin(); it != state->groupTags.end(); it++) {
        outFile << it->first << "\n";
    }
    outFile << "</groupHandles>\n";
    outFile << "<groupBits>\n";
    for (auto it = state->groupTags.begin(); it != state->groupTags.end(); it++) {
        outFile << it->second<< "\n";
    }
    outFile << "</groupBits>\n";
    outFile << "</groupInfo>\n";
}

void outputMolecules(ofstream &outFile, State *state) {
    outFile << "<molecules>\n";
    int len = py::len(state->molecules);
    for (int i=0; i<len; i++) {
        outFile << "<m>\n";
        py::extract<Molecule> mPy(state->molecules[i]);
        if (!mPy.check()) {
            assert(mPy.check());
        }
        Molecule m = mPy;
        for (int id : m.ids) {
            outFile << id << " ";
        }
        outFile << "</m>\n";
    }
    outFile << "</molecules>\n";

}

void writeXMLfileBase64(State *state, string fnFinal, int64_t turn, bool oneFilePerWrite, uint groupBit) {
    vector<Atom> &atoms = state->atoms;
    ofstream outFile;
    Bounds b = state->bounds;
    if (oneFilePerWrite) {
        outFile.open(fnFinal.c_str(), ofstream::out);
        outFile << "<data>" << endl;
    } else {
        outFile.open(fnFinal.c_str(), ofstream::app);
    }
    char buffer[BUFFERLEN];
    int ndims = state->is2d ? 2 : 3;    
    double dims[6];
    //double dims[12];
    * (Vector *)dims = b.lo;
    * (((Vector *)dims) + 1) = b.lo + b.rectComponents;
    //for (int i=1; i<4; i++) {
    //    * (((Vector *)dims)+i) = b.sides[i-1];
    //}
    
    string b64[6];
    for (int i=0; i<6; i++) {
        b64[i] = base64_encode((const unsigned char *) (dims + i), 8);
    }
    sprintf(buffer, "<configuration turn=\"%" PRId64 "\" numAtoms=\"%d\" dimension=\"%d\" periodic=\"%d%d%d\" rCut=\"%f\" padding=\"%f\" dt=\"%f\">\n", turn, (int) atoms.size(), ndims, state->periodic[0], state->periodic[1], state->periodic[2], state->rCut, state->padding, state->dt);
    outFile << buffer;
    //sprintf(buffer, "<bounds base64=\"1\" xlo=\"%s\" ylo=\"%s\" zlo=\"%s\" sxx=\"%s\" sxy=\"%s\" sxz=\"%s\" syx=\"%s\" syy=\"%s\" syz=\"%s\" szx=\"%s\" szy=\"%s\" szz=\"%s\"/>\n", b64[0].c_str(), b64[1].c_str(), b64[2].c_str(), b64[3].c_str(), b64[4].c_str(), b64[5].c_str(), b64[6].c_str(), b64[7].c_str(), b64[8].c_str(), b64[9].c_str(), b64[10].c_str(), b64[11].c_str());
    sprintf(buffer, "<bounds base64=\"1\" xlo=\"%s\" ylo=\"%s\" zlo=\"%s\" xhi=\"%s\" yhi=\"%s\" zhi=\"%s\" />\n", b64[0].c_str(), b64[1].c_str(), b64[2].c_str(), b64[3].c_str(), b64[4].c_str(), b64[5].c_str());
    outFile << buffer;
    writeAtomParams(outFile, state->atomParams);
    writeXMLChunkBase64<Atom, Vector>(outFile, atoms, "position", [] (Atom &a) {
            return a.pos;    
            }
            );

    writeXMLChunkBase64<Atom, Vector> (outFile, atoms, "velocity", [] (Atom &a) {
            return a.vel;
            }
            );
    writeXMLChunkBase64<Atom, Vector> (outFile, atoms, "force", [] (Atom &a) {
            return a.force;
            }
            );

    writeXMLChunkBase64<Atom, uint>(outFile, atoms, "groupTag", [] (Atom &a) {
            return a.groupTag;
            }
            );
    writeXMLChunkBase64<Atom, int>(outFile, atoms, "type", [] (Atom &a) {
            return a.type;
            }
            );
    writeXMLChunkBase64<Atom, int>(outFile, atoms, "id", [] (Atom &a) {
            return a.id;
            }
            );
    writeXMLChunkBase64<Atom, double>(outFile, atoms, "q", [] (Atom &a) {
            return a.q;
            }
            );

    outputGroups(outFile, state);
    outputMolecules(outFile, state);
    sprintf(buffer, "</configuration>\n");
    outFile << buffer;
    if (oneFilePerWrite) {
        outFile << "</data>" << endl;
    }
    outFile.close();


}


void writeLAMMPSTRJFile(State *state, string fn, int64_t turn, bool oneFilePerWrite, uint groupBit) {
    vector<Atom> &atoms = state->atoms;
    AtomParams &params = state->atomParams;
    int count = 0;
    
    ofstream outFile;
    if (oneFilePerWrite) {
        outFile.open(fn.c_str(), ofstream::out);
    } else {
        outFile.open(fn.c_str(), ofstream::app);
    }
	if (groupBit == 1) {
        count = atoms.size();
	} else {
		for (Atom &a : atoms) {
			if (a.groupTag & groupBit) {
				count ++;
			}
		}
	}

    // WRITE THE HEADER INFORMATION
    outFile << "ITEM: TIMESTEP" << endl << turn << endl;    // TIMESTEP
    outFile << "ITEM: NUMBER OF ATOMS" << endl << count << endl;   // NUMBER OF ATOMS
    outFile << "ITEM: BOX BOUNDS pp pp pp" << endl;
    outFile << state->bounds.lo[0] << " " << state->bounds.lo[0] + state->bounds.rectComponents[0] << endl;
    outFile << state->bounds.lo[1] << " " << state->bounds.lo[1] + state->bounds.rectComponents[1] << endl;
    outFile << state->bounds.lo[2] << " " << state->bounds.lo[2] + state->bounds.rectComponents[2] << endl;       // BOX DIMENSION

    // WRITE THE ATOM rectINFORMATION
    outFile << "ITEM: ATOMS id type x y z" << endl;         // ATOM HEADER
    for (Atom &a : atoms) {
        if (a.groupTag & groupBit) {
            outFile << a.id << " " << a.type << " " << a.pos[0] << " " << a.pos[1] << " " << a.pos[2] << endl;
        }
    }

    outFile.close();
}

void writeXYZFile(State *state, string fn, int64_t turn, bool oneFilePerWrite, uint groupBit) {
    vector<Atom> &atoms = state->atoms;
    AtomParams &params = state->atomParams;
    bool useAtomicNums = true;
    for (int atomicNum : params.atomicNums) {
        if (atomicNum == -1) {
            useAtomicNums = false;
        }
    }
    ofstream outFile;
    if (oneFilePerWrite) {
        outFile.open(fn.c_str(), ofstream::out);
    } else {
        outFile.open(fn.c_str(), ofstream::app);
    }
	if (groupBit == 1) {
		outFile << atoms.size() <<  endl << "bounds lo " << state->bounds.lo << " hi " << (state->bounds.lo + state->bounds.rectComponents);
	} else {
		int count = 0;
		for (Atom &a : atoms) {
			if (a.groupTag & groupBit) {
				count ++;
			}
		}
		outFile << count <<  endl << "bounds lo " << state->bounds.lo << " hi " << (state->bounds.lo + state->bounds.rectComponents);
	}
    for (Atom &a : atoms) {
		if (a.groupTag & groupBit) {
			int atomicNum;
			if (useAtomicNums) {
				atomicNum = params.atomicNums[a.type];
			} else {
				atomicNum = a.type;
			}
			outFile << endl << atomicNum << " " << a.pos[0] << " " << a.pos[1] << " " << a.pos[2];
		}
    }
    outFile << endl;
    outFile.close();
}

void writeXMLfile(State *state, string fnFinal, int64_t turn, bool oneFilePerWrite, uint groupBit) {
    vector<Atom> &atoms = state->atoms;
    ofstream outFile;
    Bounds b = state->bounds;
    if (oneFilePerWrite) {
        outFile.open(fnFinal.c_str(), ofstream::out);
        outFile << "<data>" << endl;
    } else {
        outFile.open(fnFinal.c_str(), ofstream::app);
    }
    char buffer[BUFFERLEN];
    int ndims = state->is2d ? 2 : 3;    
  //  double s[9];
  //  for (int i=0; i<3; i++) {
  //      for (int j=0; j<3; j++) {
  //          s[i*3 + j] = (double) b.sides[i][j];
  //      }
  //  }
    sprintf(buffer, "<configuration turn=\"%" PRId64 "\" numAtoms=\"%d\" dimension=\"%d\" periodic=\"%d%d%d\" rCut=\"%f\" padding=\"%f\" dt=\"%f\">\n", turn, (int) atoms.size(), ndims, state->periodic[0], state->periodic[1], state->periodic[2], state->rCut, state->padding, state->dt);
    outFile << buffer;
    Vector hi = b.lo + b.rectComponents;
  //  sprintf(buffer, "<bounds base64=\"0\" xlo=\"%f\" ylo=\"%f\" zlo=\"%f\" sxx=\"%f\" sxy=\"%f\" sxz=\"%f\" syx=\"%f\" syy=\"%f\" syz=\"%f\" szx=\"%f\" szy=\"%f\" szz=\"%f\"/>\n", (double) b.lo[0], (double) b.lo[1], (double) b.lo[2], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
    sprintf(buffer, "<bounds base64=\"0\" xlo=\"%f\" ylo=\"%f\" zlo=\"%f\" xhi=\"%f\" yhi=\"%f\" zhi=\"%f\" />\n", b.lo[0], b.lo[1], b.lo[2], hi[0], hi[1], hi[2]);
    outFile << buffer;
    writeAtomParams(outFile, state->atomParams);
    //going to store atom params as 
    //<atom_params num_types="x" force_type="type">
    //<handle> ... </handle>
    //<sigma>2d array mapped to 1d </sigma>
    //<epsilon> ... </epsilon>
    //<mass> ... </mass>
    //</atom_params>

    //looping over fixes                                                                                                                                    
    outFile << "<fixes>\n";
    for(Fix *f : state->fixes) {
      char buffer[BUFFERLEN];
      sprintf(buffer, "<fix type=\"%s\" handle=\"%s\">", f->type.c_str(), f->handle.c_str());
      outFile << buffer << "\n";
      outFile << f->restartChunk("xml");
      outFile << "</fix>\n";
    }
    outFile << "</fixes>\n";


    writeXMLChunk<Atom>(outFile, atoms, "position", [] (Atom &a, char buffer[BUFFERLEN]) {
            Vector pos = a.pos; sprintf(buffer, "%f %f %f\n", (double) pos[0], (double) pos[1], (double) pos[2]);
            }
            );

    writeXMLChunk<Atom>(outFile, atoms, "velocity", [] (Atom &a, char buffer[BUFFERLEN]) {
            Vector vel = a.vel; sprintf(buffer, "%f %f %f\n", (double) vel[0], (double) vel[1], (double) vel[2]);
            }
            );
    writeXMLChunk<Atom>(outFile, atoms, "force", [] (Atom &a, char buffer[BUFFERLEN]) {
            Vector force = a.force; sprintf(buffer, "%f %f %f\n", (double) force[0], (double) force[1], (double) force[2]);
            }
            );

    writeXMLChunk<Atom>(outFile, atoms, "groupTag", [] (Atom &a, char buffer[BUFFERLEN]) {
            sprintf(buffer, "%u\n", a.groupTag);
            }
            );
    writeXMLChunk<Atom>(outFile, atoms, "type", [] (Atom &a, char buffer[BUFFERLEN]) {
            sprintf(buffer, "%d\n", a.type);
            }
            );
    writeXMLChunk<Atom>(outFile, atoms, "id", [] (Atom &a, char buffer[BUFFERLEN]) {
            sprintf(buffer, "%d\n", a.id);
            }
            );
    writeXMLChunk<Atom>(outFile, atoms, "q", [] (Atom &a, char buffer[BUFFERLEN]) {
            sprintf(buffer, "%f\n", a.q);
            }
            );
    outputGroups(outFile, state);
    outputMolecules(outFile, state);
    sprintf(buffer, "</configuration>\n");
    outFile << buffer;
    if (oneFilePerWrite) {
        outFile << "/<data>" << endl;
    }
    outFile.close();

    /*
       Vector lo = state->bounds->lo;
       Vector hi = state->bounds->hi;
    FILE *f = fopen(fnFinal, "a+");
    fprintf(f, "Turn %d\n", state->turn);
    fprintf(f, "Bounds %f %f %f %f %f %f\n", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    fprintf(f, "Atoms\n");
    for (Atom &a : state->atoms) {
        fprintf(f, "%d %d %f %f %f\n", a.id, a.type, a.pos[0], a.pos[1], a.pos[2]);
    }
    fprintf(f, "end Atoms\n");
    fprintf(f, "Bonds\n");
    for (Bond &b : state->bonds) {
        fprintf(f, "%d %d %f %f %f %f %f %f\n", b.atoms[0]->id, b.atoms[1]->id, b.atoms[0]->pos[0], b.atoms[0]->pos[1], b.atoms[0]->pos[2], b.atoms[1]->pos[0], b.atoms[1]->pos[1], b.atoms[1]->pos[2]);
    }
    fprintf(f, "end Bonds\n");
    fclose(f);
*/
}




string WriteConfig::getCurrentFn(int64_t turn) {
    char buffer[200];
    if (format == "base64") {
        sprintf(buffer, "%s.xml", fn.c_str());
    } else if (format == "xyz" ) {
        sprintf(buffer, "%s.xyz", fn.c_str());
    } else if (format == "lammpstrj" ) {
        sprintf(buffer, "%s.lammpstrj", fn.c_str());
    } else {
        sprintf(buffer, "%s.xml", fn.c_str());
    }

    if (oneFilePerWrite) {
        string asStr = string(buffer);
        string turnStr = to_string(turn);
        size_t pos = asStr.find("*");
        assert(pos != string::npos);
        string finalFn = asStr.substr(0, pos) + turnStr + asStr.substr(pos+1, asStr.size());
        return finalFn;



    }
    return string(buffer);
}

WriteConfig::WriteConfig(SHARED(State) state_, string fn_, string handle_, string format_, int writeEvery_, string groupHandle_, bool unwrapMolecules_) : state(state_.get()), fn(fn_), handle(handle_), format(format_), writeEvery(writeEvery_), groupHandle(groupHandle_), unwrapMolecules(unwrapMolecules_) {
	groupBit = state->groupTagFromHandle(groupHandle);
    if (format == "base64") {
        writeFormat = &writeXMLfileBase64;
        isXML = true;
    } else if (format == "xyz") {
        writeFormat = &writeXYZFile;
        isXML = false;
    } else if (format == "lammpstrj") {
        writeFormat = &writeLAMMPSTRJFile;
        isXML = false;
    } else {
        writeFormat = &writeXMLfile;
        isXML = true;
    }
    if (fn.find("*") != string::npos) {
        oneFilePerWrite = true;
    } else {
        oneFilePerWrite = false;
        string fn = getCurrentFn(0);
        unlink(fn.c_str());
        if (isXML) {
            ofstream outFile;
            outFile.open(fn.c_str(), ofstream::app);
            outFile << "<data>\n";
        }
    }
}

void WriteConfig::finish() {
    if (isXML and not oneFilePerWrite) {
        ofstream outFile;
        outFile.open(getCurrentFn(0), ofstream::app);
        outFile << "</data>";
    }
}


void WriteConfig::write(int64_t turn) {
    if (unwrapMolecules) {
        state->unwrapMolecules();
    }
    writeFormat(state, getCurrentFn(turn), turn, oneFilePerWrite, groupBit);
}
void WriteConfig::writePy() {
    state->atomParams.guessAtomicNumbers();
    if (unwrapMolecules) {
        state->unwrapMolecules();
    }
    writeFormat(state, getCurrentFn(state->turn), state->turn, oneFilePerWrite, groupBit);
}


void export_WriteConfig() {
    py::class_<WriteConfig,
                          SHARED(WriteConfig) >("WriteConfig", py::init<SHARED(State), string, string, string, int, py::optional<string, bool> >(py::args("fn", "handle", "format", "writeEvery", "groupHandle", "unwrapMolecules"))
    )
    .def_readwrite("writeEvery", &WriteConfig::writeEvery)
    .def_readwrite("unwrapMolecules", &WriteConfig::unwrapMolecules)
    .def_readonly("handle", &WriteConfig::handle)
    .def("write", &WriteConfig::writePy)
    ;
}
