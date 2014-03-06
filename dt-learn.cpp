#include<iostream>
#include<iomanip>
#include<string>
#include<limits>
#include<map>
#include<algorithm>
#include<stdlib.h>
#include<math.h>
#include<vector>
#include<sstream>
#include<fstream>
using namespace std;
struct LabVal { // To get the value and corresponding label
	double value;
	int origType;
	int type; // 0 - Positive, 1 - Negative, 2 - Mixed

};
bool sortByVal (const LabVal* lhs,const LabVal* rhs) {
	if(lhs->value < rhs->value) {
		return true;
	}
	return false;
}
double stringToDouble(string s) {
	stringstream ss(s);
	double d;
	ss >> d;
	return d;
}

string doubleToString(double d) {
	stringstream ss;
	ss << d;
	return ss.str();
}

vector<string> tokenize(string line, char delim) {
	stringstream ss(line);
	string s;
	vector<string> tokens;
	while( getline(ss, s, delim)) {
		tokens.push_back(s);
	}
	return tokens;
}

class Feature {
	string name;
	int index;
	// 0 - Feature ; 1 - Discrete ; 2 - Real

	public:
	Feature(string n, int i) {name = n; index = i;}
	string getName() { return name; }
	int getIndex() { return index;}
	virtual void print() {
		cout << "Name: " << name << " Index: " << index;
	}
	virtual string getFirstValue() { cerr << "Error : 2345 " << endl; return "";}
	virtual int getType() { return 0; }
	virtual vector<string> getValues() { cerr << "Error : 6780 " << name << endl; return vector<string>();}
	virtual int getIndex(string n) { cerr<< "Error 9744" << endl; return -1;}
};

class DiscreteF:public Feature {
	vector<string> values;
	public:
	DiscreteF(string n, int i, vector<string>& v):Feature(n,i) {values = v;}
	int getType() { return 1; }
	vector<string> getValues() {
		return values;
	}
	int getIndex(string val) {
		for(int i=0; i<values.size(); i++) {
			if( val == values[i]) {
				return i;
			}
		}
		return -1;
	}
	string getFirstValue() {
		return values[0];
	}	
	void print() {
		Feature::print();
		cout << "{ ";
		for(int i = 0; i < values.size(); i++) {
			cout << values[i] << ", ";
		}
		cout << " }";
	}
};

class RealF : public Feature {
	public:
	RealF(string n, int i):Feature(n,i) {}
	int getType() { return 2;}
	void print() { 
		Feature::print();
		cout << "Real"; 
	}
};

class Value {
	double dblVal;
	string strVal;
	bool type; // 0 - Real; 1 - Discrete
	public:
	Value(string val) {setStrVal(val);}
	Value(double val) {setDblVal(val);}
	void setType(bool t) { type = t; }
	void setDblVal(double val) { dblVal = val; setType(0); }
	void setStrVal(string val) { strVal = val; setType(1); }
	bool getType() { return type; }
	double getDblVal() { return dblVal; }
	string getStrVal() { return strVal; }
	bool isEqual(string val) {
		if( val == strVal) {
			return true;
		}
		return false;
	}
	void print() {
		if( type == 0) {
			cout << dblVal;
		} else {
			cout << strVal;
		}
	}
};

class Instance {
	vector<Value*> values;
	bool type; // 0 - Train ; 1 - Test
	public:
	Instance() { type = 0;}
	Instance(vector<Value*>& vals ) {values = vals;}
	Value* getValue(int fIdx) {
		return values[fIdx];
	}
	string getLabel() { return values[values.size()-1]->getStrVal(); }
	bool getType() { return type; }
	void addValue(string val) {
		values.push_back(new Value(val));
	}
	void addValue(double val) {
		values.push_back(new Value(val));
	}
	void print() {
		for(int i=0; i<values.size(); i++) {
			values[i]->print();
			cout << " ";
		}
		cout << endl;
	}
	string getSpaceSeparated() {
		string strSep="";
		for(int i=0; i<values.size()-1; i++) {
			if(values[i]->getType() == 1) {
				strSep += values[i]->getStrVal();
				strSep += " ";	
			} else {
				strSep += doubleToString(values[i]->getDblVal());
				strSep += " ";
			}
		}
		return strSep;
	}
};

class Dataset {
	vector<Feature*> features;
	vector<Instance*> instances;
	public:
	Dataset() {}
	int numFeatures() {return features.size();}
	int numInstances() {return instances.size();}

	Feature* getFeature(int idx) { return features[idx];}
	Instance* getInstance(int idx) { return instances[idx];}

	string getFirstLabel() {
		return features[features.size()-1]->getFirstValue(); 
	}
	void addFeature(string line) {
		vector<string> tokens = tokenize(line, ' ');
		if( tokens[2] == "real") {
			// Real valued feature
			features.push_back(new RealF(tokens[1].substr(1, tokens[1].length()-2), features.size()));

		} else {
			string name = tokens[1];
			tokens.erase(tokens.begin(), tokens.begin() + 3);
			for(int i = 0; i < tokens.size(); i++) {
				tokens[i] = tokens[i].substr(0, tokens[i].length() - 1);
			} 
			features.push_back(new DiscreteF(name.substr(1, name.length()-2), features.size(), tokens));
			// Discrete valued feature
		}
	}

	void printFeatures() {
		for(int i = 0; i< features.size(); i++) {
			features[i]->print();
			cout << endl;
		}
	}
	void printInstances() {
		for(int i = 0; i<instances.size(); i++) {
			instances[i]->print();
		}
	}
	void addInstance(string line) {
		vector<string> tokens = tokenize(line, ',');
		if(features.size() != tokens.size()) {
			cerr << "Size mismatch " << features.size() << " " << tokens.size() << endl;
			exit(0);
		}
		instances.push_back(new Instance);
		for(int i = 0; i < tokens.size(); i++) {
			if(features[i]->getType() == 1) {

				(instances.back())->addValue(tokens[i]);
			} else {
				(instances.back())->addValue(stringToDouble(tokens[i]));

			}
		}

	}
};


class ID3Node {
	vector<ID3Node*> children;
	int decFeature; // deciding feature's index
	double threshold;
	int m;
	double entropy; // entropy in this node
	string prediction; // Only if it is a leaf
	bool isLeaf;
	map<int, double> entropyMap; // To store entropy of Real features
	map<int, double> threshMap; // To store threshold of Real features
	Dataset* ds;
	int numPos, numNeg;
	vector<int> fIdx;
	vector<int> iIdx;


	public:
	ID3Node(Dataset* d, int _m) {
		ds = d;
		m = _m;
		decFeature = -1;
		isLeaf = false;
	}
	bool getLeaf() {
		return isLeaf;
	}
	ID3Node(ID3Node& cand) {
		entropy = cand.entropy;
		m = cand.m;
		children = cand.children;
		decFeature = cand.decFeature;
		threshold = cand.threshold;
		prediction = cand.prediction;
		isLeaf = cand.isLeaf;
		entropyMap = cand.entropyMap;
		threshMap = cand.threshMap;
		ds = cand.ds;
		fIdx = cand.fIdx;
		iIdx = cand.iIdx;
		numPos = cand.numPos;
		numNeg = cand.numNeg;
	}
	string classify(Instance* inst) {
		if(isLeaf) {
			return prediction;
		}
		Feature* f = ds->getFeature(fIdx[decFeature]);
		Value* val = inst->getValue(fIdx[decFeature]);
		if( f->getType() == 1) {
			string v = val->getStrVal();
			return (children[f->getIndex(v)])->classify(inst);
		} else {
			double v = val->getDblVal();
			if( v <= threshold) {
				return children[0]->classify(inst);
			} else {
				return children[1]->classify(inst);
			}
		}
	}
	void initFeatures(vector<int> fIndex, int index) {
		// Copy all fIndex except index
		for(int i=0; i < fIndex.size(); i++) {
			if(fIndex[i] != index) {
				fIdx.push_back(fIndex[i]);
			}
		}
	}
 	void print(string prefix = "") {
		if(isLeaf) {
			return;
		}
		Feature* f = ds->getFeature(fIdx[decFeature]);	
		if(f->getType() == 1 ) {
			vector<string> vals = f->getValues();
			for(int j = 0; j < vals.size(); j++) {
				cout << prefix << f->getName() << " = " << vals[j] << " [" << children[j]->numNeg << " " << children[j]->numPos << "]";
				if(children[j]->getLeaf()) {
					cout << ": " << children[j]->prediction << endl;
				} else {
					cout << endl;
					children[j]->print(prefix + "|\t");
				}
			}
		} else {
			cout << prefix << f->getName();
			cout << " <= " << setprecision(6) << threshold << " [" << children[0]->numNeg << " " << children[0]->numPos << "]";	
			if(children[0]->getLeaf()) {
				cout << ": " << children[0]->prediction << endl;
			} else {
				cout << endl;
				children[0]->print(prefix + "|\t");
			}
			cout << prefix << f->getName();
			cout << " > " << setprecision(6) << threshold << " [" << children[1]->numNeg << " " << children[1]->numPos << "]";	
			if(children[1]->getLeaf()) {
				cout << ": " << children[1]->prediction << endl;
			} else {
				cout << endl;
				children[1]->print(prefix + "|\t");
			}
		}
	}
	void makeLeaf() {
		isLeaf = true;
		string label;
		// set prediction
		numPos = 0, numNeg = 0;
		for(int i=0; i < iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			label = inst->getLabel();
			if( label == "positive") {
				numPos++;
			} else {
				numNeg++;
			}
		}
		if( numPos > numNeg) {
			prediction = "positive";
		} else if( numNeg > numPos) {
			prediction = "negative";
		} else {
			prediction = ds->getFirstLabel();
		}


	}
	void addFeatureIdx(int index) {
		fIdx.push_back(index);
	}
	void addInstanceIdx(int index) {
		iIdx.push_back(index);
	}
	void makeRoot() {
		// make all feauture active
		for(int i = 0; i < ds->numFeatures(); i++) {
			fIdx.push_back(i);
		}
		// make all instances active
		for(int i = 0; i < ds->numInstances(); i++) {
			iIdx.push_back(i);
		}
		findMyEntropy();
		//cout << "UUUU Entropy : " << entropy << endl;
	}
	string getPrediction() { return prediction; }
	bool stop() { 
		string oldLabel, label;
		if(iIdx.size() < m) {
			//cout << "UUU iIdx.size() m " << iIdx.size() << " " << m << endl;
			return true;
		}
		if(fIdx.size() == 1) {
			//cout << "UUU fIdx.size() " << fIdx.size()  << endl;
			return true;
		}
		Instance* inst = ds->getInstance(iIdx[0]);
		oldLabel = inst->getLabel();
		for(int i=1; i < iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			label = inst->getLabel();
			if( label != oldLabel) {
				return false;
			}
		}
		//cout << "UUU All same " << endl;
		return true;
	}

	vector<ID3Node*> splitDiscrete(int fIndex) {
		Feature* f = ds->getFeature(fIdx[fIndex]);
		vector<ID3Node*> splits;
		vector<string> values = f->getValues();	
		for( int i = 0; i< values.size(); i++) {	
			ID3Node* node = new ID3Node(ds, m);
			// Add all features except current
			node->initFeatures(fIdx, fIdx[fIndex]);	
			// Add all instances whose values match
			for(int j=0; j<iIdx.size(); j++) {
				Instance* inst = ds->getInstance(iIdx[j]);
				if( inst->getValue(fIdx[fIndex])->isEqual(values[i])) {
					node->addInstanceIdx(iIdx[j]);
				}
			}
			node->findMyEntropy();
			splits.push_back(node);
		}	
		return splits;
	}

	vector<ID3Node*> bestSplitReal(int fIndex, double& minEntropy, double& bestThresh) {
		Feature* f = ds->getFeature(fIdx[fIndex]);

		vector<LabVal*> values;
		for(int i=0; i<iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			string label = inst->getLabel();
			LabVal* lv = new LabVal();
			lv->value = inst->getValue(f->getIndex())->getDblVal();
			if( label == "positive") {
				lv->origType = 0;
				lv->type = 0;
			} else {
				lv->origType = 1;
				lv->type = 1;
			}
			values.push_back(lv);
		}
		//cout << "UUUU values size" << values.size() << endl;
		sort(values.begin(), values.end(),sortByVal);
		//for(int i = 0; i < values.size(); i++) {
		//	cout << "UUUU val " << values[i]->value << endl; 
		//}
		//exit(0);
		// Now, assigning the correct types
		for(int i = 1; i < values.size(); i++) {
			if( values[i]->value == values[i-1]->value) {
				if( values[i]->origType != values[i-1]->origType) {
					values[i]->type = 2;
					values[i-1]->type = 2;
				}
			}
		}
		vector<double> candSplits;

		for(int i = 1; i < values.size(); i++) {
			if( values[i]->value != values[i-1]->value) {
				if(values[i]->type != values[i-1]->type || values[i]->type == 2 || values[i-1]->type == 2) {
					candSplits.push_back((values[i]->value + values[i-1]->value)/2.0);
					//cout <<"UUUU " << f->getName() << " " << candSplits.back() << endl;
				}
			}
		}

		minEntropy = numeric_limits<double>::max( );
		for( int i=0; i< candSplits.size(); i++) {
			double thisEntropy = findEntropy(candSplits[i], fIndex);
			//cout << "UUUU	thisEntr " << thisEntropy << endl; 
			if( thisEntropy < minEntropy) {
				minEntropy = thisEntropy;
				bestThresh = candSplits[i];
			}
		}		
		//cout << "UUUU minEntropy " << minEntropy << endl;
		vector<ID3Node*> bestSplitNode;
		bestSplitNode.push_back(new ID3Node(ds, m));
		bestSplitNode.push_back(new ID3Node(ds, m));
		bestSplitNode[0]->initFeatures(fIdx, -1);
		bestSplitNode[1]->initFeatures(fIdx, -1);
		for(int i = 0; i<iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			double value = inst->getValue(f->getIndex())->getDblVal();
			if( value <= bestThresh) {
				bestSplitNode[0]->addInstanceIdx(iIdx[i]);
			} else {
				bestSplitNode[1]->addInstanceIdx(iIdx[i]);
			}
		}
		bestSplitNode[0]->findMyEntropy();
		bestSplitNode[1]->findMyEntropy();
		return bestSplitNode;	
	} 
	// Entropy for discrete features
	double findEntropy(int fIndex) {
		int numPosLeft = 0, numNegLeft = 0, numPosRt = 0, numNegRt = 0;
		vector<string> values = ds->getFeature(fIdx[fIndex])->getValues();
		//cout << "UUUU values.size " << values.size()  << endl;
		vector<int> posCnt(values.size(),0);
		vector<int> negCnt(values.size(),0);
		map<string, int> valMap;
		for(int i = 0; i<values.size(); i++) {
			valMap[values[i]] = i;
		}
		for(int i = 0; i < iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			string label = inst->getLabel();
			string value = inst->getValue(fIdx[fIndex])->getStrVal();
			//cout << "UUUU values size" << values.size() << endl;
			int valIdx = valMap[value];
			//cout << "UUUU valIdx " << valIdx << " value " << value << endl;
			if( label == "positive") {
				posCnt[valIdx]++;
			} else {
				negCnt[valIdx]++;
			}		 	
		}
		double totEntropy = 0;
		for(int i=0; i < values.size(); i++) {
			//cout << "Values[i]: " << values[i] << " Num pos : " << posCnt[i] << " " << "Num neg : " << negCnt[i] << endl;
			if( posCnt[i] == 0 || negCnt[i] == 0) {
				continue;
			}
			double tot = (double) (posCnt[i]+negCnt[i]);
			double p1 = posCnt[i]/ tot;
			double p2 = negCnt[i]/tot;
			totEntropy += ((-p1)*log10(p1) + (-p2)*log10(p2))*tot/iIdx.size();
		}
		return totEntropy;
	}
	// Find entropy of real features
	double findEntropy(double _thresh, int fIndex) {
		int numPosLeft = 0, numNegLeft = 0, numPosRt = 0, numNegRt = 0;
		for(int i=0; i<iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			string label = inst->getLabel();
			double value = inst->getValue(fIdx[fIndex])->getDblVal();
			if( value <= _thresh) {
				if( label == "positive") {
					numPosLeft++;
				} else {
					numNegLeft++;
				}
			} else {
				if( label == "positive") {
					numPosRt++;
				} else {
					numNegRt++;
				}
			}
		}
		double numTotLeft = (double)(numPosLeft + numNegLeft);
		double numTotRt = (double)(numPosRt + numNegRt);
		double p1L = numPosLeft/numTotLeft;
		double p2L = numNegLeft/numTotLeft;
		double p1R = numPosRt/numTotRt;
		double p2R = numNegRt/numTotRt;
		double entropyL;
		if(numPosLeft == 0 || numNegLeft == 0) {
			entropyL = 0;
		} else {
			entropyL = (-p1L)* log10(p1L) + (-p2L) *log10(p2L);
		}
		double entropyR;
		if(numPosRt == 0 || numNegRt == 0) {
			entropyR = 0;
		} else {
			entropyR = (-p1R)* log10(p1R) + (-p2R)* log10(p2R);
		}
		//cout << "UUUU NumPosLeft, NumNegLeft, NumPosRt, NumNegRt " << numPosLeft << " " << numNegLeft << " " << numPosRt << " " << numNegRt << endl; 
		double thisEntropy = (numTotLeft/iIdx.size())*entropyL + (numTotRt/iIdx.size())*entropyR;
		//cout << "UUUU This feature, threshold, entropy " << (ds->getFeature(fIdx[fIndex]))->getName() << " " << _thresh << " " << thisEntropy << endl;
		return thisEntropy;
	}
	void findMyEntropy() {
		numPos = 0, numNeg = 0; 
		for(int i=0; i<iIdx.size(); i++) {
			Instance* inst = ds->getInstance(iIdx[i]);
			string label = inst->getLabel();
			if( label == "positive") {
				numPos++;
			} else {
				numNeg++;
			}
		}
		if(numPos == 0 || numNeg == 0 ) {
			entropy = 0;
			return;
		}
		//cout << "UUUU Num pos : " << numPos << endl;
		//cout << "UUUU Num neg : " << numNeg << endl;
		double p1 = (double)numPos/(numPos+numNeg);
		double p2 = (double) numNeg/(numPos+numNeg);
		//cout << "UUUU p1: " << p1 << endl;
		//cout << "UUUU p2: " << p2 << endl;
		entropy = (-p1)*log10(p1) + (-p2)* log10(p2);

	}
	vector<vector<ID3Node*> > chooseCandidateSplits() {
		//cout << "UUUU chooseCandidateSplits " << endl;
		vector<vector<ID3Node*> > candSplits;
		double _entropy;
		double thresh;
		for(int i=0; i < fIdx.size()-1; i++) {
			Feature* currF = ds->getFeature(fIdx[i]);
			if(!currF) {
				cerr << "Error 3476" << endl;
				exit(0);
			}
			int type = currF->getType();
			vector<ID3Node*> splits; 
			if( type == 1) {
				// Discrete Feature
				splits = splitDiscrete(i);
			} else if( type == 2) {
				splits = bestSplitReal(i, _entropy, thresh);
				entropyMap[fIdx[i]] = _entropy;
				threshMap[fIdx[i]] = thresh;
			} else {
				cerr << "Error 3457" << endl;
			}
			candSplits.push_back(splits);	
		}
		return candSplits;
	}
	double getEntropy() {
		return entropy;
	}

	double findBestSplit(vector<vector<ID3Node*> > cand) {
		//cout << "UUUU findBestSplit " << endl;
		double minEntropy = numeric_limits<double>::max( );
		int bestSplit;
		double thisEntropy;
		//cout << "UUUU cand size " << cand.size() << endl;
		// populate children and delete unselected candidates
		for( int i=0; i< fIdx.size()-1; i++) {
			Feature* currF = ds->getFeature(fIdx[i]);
			int type = currF->getType();
			if( type == 1) {
				// Discrete Feature
				thisEntropy = findEntropy(i);
			} else if( type == 2) {
				thisEntropy = entropyMap[fIdx[i]];
			} else {
				cerr << "Error 1234" << endl;
			}
			//cout << "UUUU feature, entropy : " << currF->getName() << " " << thisEntropy << endl;
			if( thisEntropy < minEntropy) {
				minEntropy = thisEntropy;
				//cout << "UUUU curent min feature, entropy : " << currF->getName() << " " << thisEntropy << endl;
				bestSplit = i;
				threshold = threshMap[fIdx[i]];
			}
		}
		decFeature = bestSplit;
		//cout << "UUUU Dec feature: " << ds->getFeature(fIdx[decFeature])->getName() << " thresh " << threshold << endl;
		//cout << endl;
		for(int i=0; i<cand.size(); i++) {
			if( i == bestSplit) {
				//cout << "UUUU cand[12] size " << cand[i].size() << endl;
				for(int j = 0; j < cand[i].size(); j++) {
					children.push_back(new ID3Node(*cand[i][j]));
				}
			}
		}
		cand.clear();
		//cout << "UUUU children size: " << children.size() << endl;
		return minEntropy;
	}

	void printFeatures() {
		for(int i = 0; i < fIdx.size(); i++) {
			cout << (ds->getFeature(fIdx[i]))->getName() << " ";
		}
		cout << endl;
	}

	void makeSubtree() {
		//cout << "UUUU makeSubTree" << endl;
		if(stop()) {
			//cout << "UUUU stopping" << endl;
			makeLeaf();
			return;
		}
		vector<vector<ID3Node*> > candidates = chooseCandidateSplits();
		double minEntropy = findBestSplit(candidates); // populate children and delete unselected candidates
		//cout << "UUU minEntropy entropy " << minEntropy << " " << entropy << endl;
		if(minEntropy > entropy) {
			makeLeaf();
			return;
		}
		//cout << "UUUU children size " << children.size() << endl;
		for(int i = 0; i < children.size(); i++) {
			children[i]->makeSubtree();
		}
	}
};

void readFile(Dataset& ds, char* fileName) {
	ifstream file (fileName);
	string line;
	bool isData = false;
	if( file.is_open()) {
		while(getline(file, line)) {
			if( line.length() < 5) {
				continue;
			} 
			string subStr = line.substr(0, 5);
			if( subStr == "@data") {
				isData = true;
				continue;
			} 
			if( subStr == "@attr") {
				ds.addFeature(line);
				continue;
			} 
			if( subStr == "@rela") {
				continue;
			} 
			if( isData) {
				ds.addInstance(line);
			}
		}
	}
}

void part1(Dataset& testDs, ID3Node* node) {
	int numTestInst = testDs.numInstances();
	int correct = 0;
	cout << endl;
	for(int i = 0; i < numTestInst; i++) {
		Instance* inst = testDs.getInstance(i);
		string spaceSep = inst->getSpaceSeparated();
		string pred = node->classify(inst);
		string actLab = inst->getLabel();
		cout << spaceSep << " " << pred << " " << actLab << endl;
		if( pred == actLab) {
			correct++;
		} 
	}
	cout << correct << " " << numTestInst << endl;

}

int main(int argc, char* argv[]) {
	if(argc != 4) {
		cerr << "Usage : ./dt-learn <train-file> <test-file> <m>" << endl;
		exit(1);
	}
	cout << fixed;
	int m = atoi(argv[3]);
	Dataset trainDs;
	readFile(trainDs,argv[1]);
	ID3Node* node = new ID3Node(&trainDs, m);
	node->makeRoot();
	node->makeSubtree();
	node->print();


	Dataset testDs;
	readFile(testDs,argv[2]);

	part1(testDs, node);


	return 0;
}
