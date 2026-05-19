import xml.etree.ElementTree as ET
import json

def parse_weather(weather_element):
    """Parse weather element and its attributes"""
    weather_data = {
        'route_percentage': int(weather_element.get('route_percentage'))
    }
    # Convert all other attributes to float
    for key, value in weather_element.attrib.items():
        if key != 'route_percentage':
            weather_data[key] = float(value)
    return weather_data

def parse_position(position_element):
    """Parse position/waypoint element"""
    return {
        'x': float(position_element.get('x')),
        'y': float(position_element.get('y')),
        'z': float(position_element.get('z'))
    }

def parse_scenario(scenario_element):
    """Parse scenario element and all its children"""
    scenario_data = {
        'name': scenario_element.get('name'),
        'type': scenario_element.get('type')
    }
    
    # Process all child elements
    for child in scenario_element:
        if child.tag == 'trigger_point':
            # Handle trigger point coordinates
            scenario_data['trigger_point'] = {
                'x': float(child.get('x')),
                'y': float(child.get('y')),
                'z': float(child.get('z')),
                'yaw': float(child.get('yaw'))
            }
        else:
            # Handle other properties with 'value' attribute
            if 'value' in child.attrib:
                # Convert numerical values to appropriate types
                value = child.get('value')
                try:
                    # Try to convert to int first
                    converted = int(value)
                    scenario_data[child.tag] = converted
                except ValueError:
                    try:
                        # Then try float
                        converted = float(value)
                        scenario_data[child.tag] = converted
                    except ValueError:
                        # Keep as string if not numerical
                        scenario_data[child.tag] = value
    
    return scenario_data

def convert_xml_to_json(xml_file, json_file):
    """Main function to convert XML file to JSON"""
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    routes_data = []
    
    # Process each route
    for route_element in root.findall('route'):
        route_data = {
            'id': route_element.get('id'),
            'town': route_element.get('town'),
            'weathers': [],
            'waypoints': [],
            'scenarios': []
        }
        
        # Process weathers
        weathers_element = route_element.find('weathers')
        if weathers_element is not None:
            for weather_element in weathers_element.findall('weather'):
                route_data['weathers'].append(parse_weather(weather_element))
        
        # Process waypoints
        waypoints_element = route_element.find('waypoints')
        if waypoints_element is not None:
            for position_element in waypoints_element.findall('position'):
                route_data['waypoints'].append(parse_position(position_element))
        
        # Process scenarios
        scenarios_element = route_element.find('scenarios')
        if scenarios_element is not None:
            for scenario_element in scenarios_element.findall('scenario'):
                route_data['scenarios'].append(parse_scenario(scenario_element))
        
        routes_data.append(route_data)
    
    # Write to JSON file
    with open(json_file, 'w') as f:
        json.dump(routes_data, f, indent=2)

# Example usage
if __name__ == "__main__":
    input_xml = "leaderboard/data/routes_devtest.xml"
    output_json = "routes.json"
    convert_xml_to_json(input_xml, output_json)
    print(f"Conversion complete! Output saved to {output_json}")