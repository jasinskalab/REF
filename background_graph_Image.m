axis off

patch('Vertices',mesh.vertices,...
    'SpecularStrength',0.12,...
    'DiffuseStrength',0.9,...
    'AmbientStrength',0.2,...
    'Faces',mesh.faces,...
    'EdgeAlpha',0,'FaceColor',[1 1 1],...
    'FaceLighting','phong','FaceAlpha',.6);

% Set the remaining axes properties
set(gca,'CameraPosition',...
    [-78.544945992385 545.692074826472 4604.59793108604],'CameraTarget',...
    [129.6960774264 138.177489128 141.54136287528],'CameraUpVector',...
    [0.00734845636125674 0.995861375240189 -0.0905876453708009],...
    'CameraViewAngle',2.2124039229879,'DataAspectRatio',[1 1 1],...
    'LineStyleOrderIndex',50,'PlotBoxAspectRatio',...
    [1 1.18303201283536 1.08555630447355]);


% Create light
light(gca,...
    'Position',[-12.1761048068871 -21.0403042499821 -10.584375361544],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[-12.1761048068871 -21.0403042499821 285.939319434555],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[-12.1761048068871 293.129906236909 -10.584375361544],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[-12.1761048068871 293.129906236909 285.939319434555],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[268.858903682521 -21.0403042499821 -10.584375361544],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[268.858903682521 -21.0403042499821 285.939319434555],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[268.858903682521 293.129906236909 -10.584375361544],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);

% Create light
light(gca,...
    'Position',[268.858903682521 293.129906236909 285.939319434555],...
    'Style','local',...
    'Color',[0.4 0.4 0.4]);
