#include "xml_func.h"

std::vector<std::string> xml_readStrings(pugi::xml_node &parent, std::string tag) {
	auto child = parent.child(tag.c_str());
	if (child) {
        std::vector<std::string> res;
        std::istringstream ss(child.first_child().value());
        std::string s;
		while (ss >> s) {
			res.push_back(s);
		}
		return res;
	}
	return std::vector<std::string>();
}

