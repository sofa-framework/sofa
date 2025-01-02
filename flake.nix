{
  description = "SOFA is an open-source framework for interactive physics simulation, with emphasis on biomechanical and robotic simulations";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # ref. https://github.com/NixOS/nixpkgs/pull/348549
    nixpkgs.url = "github:nim65s/nixpkgs/qt6-libqglviewer";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs =
    inputs@{ flake-parts, nixgl, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', system, ... }:
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            overlays = [ nixgl.overlay ];
          };
          apps.nixgl = {
            type = "app";
            program = pkgs.writeShellApplication {
              name = "nixgl-sofa";
              text = "${pkgs.lib.getExe pkgs.nixgl.auto.nixGLDefault} ${pkgs.lib.getExe self'.packages.sofa}";
            };
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.sofa;
            sofa = pkgs.callPackage ./package.nix { };
          };
        };
    };
}
